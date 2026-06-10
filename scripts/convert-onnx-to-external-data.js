#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

const DEFAULT_PYTHON = 'python3';
// Tensors below this byte size stay embedded in the protobuf (scalars,
// shapes). Everything else moves into the .data file.
const DEFAULT_SIZE_THRESHOLD = 1024;

const MODEL_PRESETS = {
  'bardsai-fp16': {
    modelId: 'bardsai/eu-pii-anonimization-multilang',
    inputFile: path.join(
      'generated',
      'models',
      'ner',
      'bardsai-eu-pii-anonimization-multilang',
      'onnx',
      'model_fp16.onnx'
    ),
  },
};

function usage() {
  return `
Repackage an embedded-weights ONNX model as ONNX external data (graph protobuf + sidecar .data file).
The weights are bit-identical; only the storage layout changes. This keeps ONNX Runtime's WebGPU
session init from copying the whole protobuf through the never-shrinking wasm heap.

Usage:
  npm run convert:model:external-data -- [--model all|bardsai-fp16] [--force]
  node scripts/convert-onnx-to-external-data.js --input <file> [options]

Options:
  --model <name>          Preset to convert: all, ${Object.keys(MODEL_PRESETS).join(', ')}. Default: all.
  --input <file>          Embedded-weights ONNX input for a custom conversion.
  --output-file <file>    Output .onnx path. Default: the input path (in-place conversion).
  --location <name>       External-data file name recorded in the protobuf and written next to the
                          output. Default: <output basename>.data. Must match the \`path\` passed to
                          ONNX Runtime via session_options.externalData (see NerExternalDataAsset).
  --python <command>      Python command with the onnx package installed. Default: ${DEFAULT_PYTHON}
  --force                 Overwrite an existing external-data output.
  --help                  Show this help.

Preset conversions run in place under generated/models, which webpack copies into dist/models during
npm run build:ext. Do NOT use transformers.js's use_external_data_format for this: it records
underscore-style \`…onnx_data\` locations that the runtime wiring cannot address.
`.trim();
}

function parseArgs(argv) {
  const options = {
    model: 'all',
    python: DEFAULT_PYTHON,
    force: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      const value = argv[i + 1];
      if (!value || value.startsWith('--')) {
        throw new Error(`${arg} requires a value.`);
      }
      i += 1;
      return value;
    };

    switch (arg) {
      case '--model':
        options.model = next();
        break;
      case '--input':
      case '-i':
        options.inputFile = next();
        break;
      case '--output-file':
      case '-o':
        options.outputFile = next();
        break;
      case '--location':
        options.location = next();
        break;
      case '--python':
        options.python = next();
        break;
      case '--force':
        options.force = true;
        break;
      case '--help':
      case '-h':
        options.help = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  return options;
}

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function fileMetadata(filePath) {
  const stat = fs.statSync(filePath);
  return {
    path: filePath,
    bytes: stat.size,
    sha256: sha256(filePath),
  };
}

function runPythonConversion({ inputFile, outputFile, location, python }) {
  const code = `
import json
import onnx
from onnx.external_data_helper import convert_model_to_external_data

input_path = ${JSON.stringify(inputFile)}
output_path = ${JSON.stringify(outputFile)}
location = ${JSON.stringify(location)}

# load_external_data=True (the default) materializes any existing external
# tensors, so re-running on an already-converted model is safe.
model = onnx.load(input_path)
convert_model_to_external_data(
    model,
    all_tensors_to_one_file=True,
    location=location,
    size_threshold=${DEFAULT_SIZE_THRESHOLD},
    convert_attribute=False,
)
onnx.save_model(model, output_path)

# Verify every externalized tensor records exactly the location we passed —
# this string must match the runtime's session_options.externalData path.
reloaded = onnx.load(output_path, load_external_data=False)
locations = set()
for tensor in reloaded.graph.initializer:
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        locations.update(
            entry.value for entry in tensor.external_data if entry.key == "location"
        )
if locations != {location}:
    raise SystemExit(f"external-data locations {sorted(locations)} != [{location}]")

onnx.checker.check_model(output_path)
print(json.dumps({"externalTensorLocations": sorted(locations)}))
`.trim();

  const result = spawnSync(python, ['-c', code], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  if (result.status !== 0) {
    const details = (result.stderr || result.stdout || '').trim();
    throw new Error(
      [
        `Failed to convert ONNX model to external data with ${python}.`,
        'Install the onnx package in that Python environment.',
        details,
      ]
        .filter(Boolean)
        .join(' ')
    );
  }

  return JSON.parse(result.stdout.trim().split('\n').pop());
}

function convertOne(options) {
  if (!options.inputFile) {
    throw new Error('Missing --input.');
  }
  const inputFile = path.resolve(options.inputFile);
  if (!fs.existsSync(inputFile) || !fs.statSync(inputFile).isFile()) {
    throw new Error(`Missing ONNX input: ${inputFile}`);
  }

  const outputFile = path.resolve(options.outputFile || inputFile);
  const location = options.location || `${path.basename(outputFile)}.data`;
  const dataFile = path.join(path.dirname(outputFile), location);

  if (fs.existsSync(dataFile) && !options.force) {
    throw new Error(
      `External-data output already exists: ${dataFile}. Rerun with --force to overwrite.`
    );
  }

  const inputBytes = fs.statSync(inputFile).size;
  const verification = runPythonConversion({
    inputFile,
    outputFile,
    location,
    python: options.python || DEFAULT_PYTHON,
  });

  const result = {
    modelId: options.modelId,
    input: { path: inputFile, bytes: inputBytes },
    output: fileMetadata(outputFile),
    externalData: fileMetadata(dataFile),
    location,
    ...verification,
  };

  const manifestPath = path.join(
    path.dirname(path.dirname(outputFile)),
    `${path.basename(outputFile, '.onnx')}-external-data-manifest.json`
  );
  fs.writeFileSync(
    manifestPath,
    `${JSON.stringify({ generatedAt: new Date().toISOString(), ...result }, null, 2)}\n`
  );
  result.manifestPath = manifestPath;

  return result;
}

function expandJobs(options) {
  if (options.inputFile) {
    return [options];
  }

  if (options.model === 'all') {
    if (options.outputFile || options.location) {
      throw new Error('Use --model <name> with --output-file/--location, or omit them for --model all.');
    }
    return Object.values(MODEL_PRESETS).map((preset) => ({ ...options, ...preset }));
  }

  const preset = MODEL_PRESETS[options.model];
  if (!preset) {
    throw new Error(
      `Unknown --model ${options.model}. Expected one of: all, ${Object.keys(MODEL_PRESETS).join(', ')}.`
    );
  }

  return [{ ...options, ...preset }];
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const kib = bytes / 1024;
  if (kib < 1024) return `${kib.toFixed(1)} KiB`;
  return `${(kib / 1024).toFixed(1)} MiB`;
}

function printResult(result) {
  console.log(`Repackaged ONNX model as external data${result.modelId ? ` for ${result.modelId}` : ''}`);
  console.log(`Input: ${result.input.path} (${formatBytes(result.input.bytes)})`);
  console.log(`Graph: ${result.output.path} (${formatBytes(result.output.bytes)})`);
  console.log(`Weights: ${result.externalData.path} (${formatBytes(result.externalData.bytes)})`);
  console.log(`Location recorded in protobuf: ${result.location}`);
  console.log(`Manifest: ${result.manifestPath}`);
}

function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  if (options.help) {
    console.log(usage());
    return [];
  }

  const jobs = expandJobs(options);
  const results = jobs.map(convertOne);
  results.forEach(printResult);
  return results;
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = {
  DEFAULT_SIZE_THRESHOLD,
  MODEL_PRESETS,
  parseArgs,
  expandJobs,
  convertOne,
};
