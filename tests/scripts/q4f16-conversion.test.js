const fs = require('fs');
const os = require('os');
const path = require('path');

const {
  DEFAULT_BLOCK_SIZE,
  DEFAULT_BITS,
  DEFAULT_OUTPUT_ROOT,
  DEFAULT_SYMMETRIC,
  MODEL_PRESETS,
  Q4F16_MANIFEST_FILE,
  parseArgs,
  expandJobs,
  convertOne,
} = require('../../scripts/convert-source-models-to-q4f16');

function successfulPythonSpawn(outputFile) {
  const calls = [];
  const spawnSyncImpl = jest.fn((command, args) => {
    calls.push({ command, args });
    const code = args[1];
    if (code.includes('convert_float_to_float16')) {
      const match = code.match(/onnx\.save\(model_fp16, ("[^"]+")\)/);
      if (match) {
        const fp16Output = JSON.parse(match[1]);
        fs.mkdirSync(path.dirname(fp16Output), { recursive: true });
        fs.writeFileSync(fp16Output, Buffer.from('fake-intermediate-fp16'));
      }
      return { status: 0, stdout: '', stderr: '' };
    }
    if (code.includes('MatMulNBitsQuantizer')) {
      fs.mkdirSync(path.dirname(outputFile), { recursive: true });
      fs.writeFileSync(outputFile, Buffer.from('fake-q4f16'));
      fs.writeFileSync(`${outputFile}.data`, Buffer.from('fake-q4f16-weights'));
      return { status: 0, stdout: '', stderr: '' };
    }
    if (code.includes('onnx.checker.check_model')) {
      return {
        status: 0,
        stdout: JSON.stringify({
          onnxChecker: 'passed',
          matMulNBitsNodeCount: 7,
          externalTensorLocations: [`${path.basename(outputFile)}.data`],
        }),
        stderr: '',
      };
    }
    return { status: 0, stdout: '', stderr: '' };
  });
  spawnSyncImpl.calls = calls;
  return spawnSyncImpl;
}

describe('q4f16 conversion script', () => {
  let tempRoot;

  beforeEach(() => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'pg-q4f16-convert-'));
  });

  afterEach(() => {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  });

  test('parses documented options', () => {
    expect(
      parseArgs([
        '--model',
        'bardsai',
        '--source-dir',
        'source',
        '--input',
        'source/onnx/model_fp16.onnx',
        '--output-dir',
        'output',
        '--output-file',
        'output/onnx/custom.onnx',
        '--python',
        'python',
        '--block-size',
        '64',
        '--symmetric',
        '--accuracy-level',
        '2',
        '--force',
      ])
    ).toEqual(
      expect.objectContaining({
        model: 'bardsai',
        sourceDir: 'source',
        input: 'source/onnx/model_fp16.onnx',
        outputDir: 'output',
        outputFile: 'output/onnx/custom.onnx',
        python: 'python',
        blockSize: 64,
        symmetric: true,
        accuracyLevel: 2,
        force: true,
      })
    );

    expect(parseArgs(['--no-symmetric'])).toEqual(
      expect.objectContaining({ symmetric: false })
    );
  });

  test('expands the default BardsAI preset into the experimental generated output', () => {
    const jobs = expandJobs(parseArgs([]));

    expect(jobs).toHaveLength(1);
    expect(jobs[0]).toEqual(
      expect.objectContaining({
        modelId: 'bardsai/eu-pii-anonimization-multilang',
        sourceDir: path.join('.model-sources', 'bardsai-eu-pii-anonimization-multilang'),
        outputDir: path.join(DEFAULT_OUTPUT_ROOT, 'bardsai-eu-pii-anonimization-multilang'),
        blockSize: DEFAULT_BLOCK_SIZE,
        symmetric: DEFAULT_SYMMETRIC,
      })
    );
    expect(path.join(MODEL_PRESETS.bardsai.outputDir, 'onnx', 'model_q4f16.onnx')).toBe(
      path.join('generated', 'models', 'ner', 'bardsai-eu-pii-anonimization-multilang', 'onnx', 'model_q4f16.onnx')
    );
  });

  test('refuses to replace q4f16 output unless force is set', () => {
    const sourceDir = path.join(tempRoot, 'source');
    const outputDir = path.join(tempRoot, 'output');
    fs.mkdirSync(path.join(sourceDir, 'onnx'), { recursive: true });
    fs.mkdirSync(path.join(outputDir, 'onnx'), { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'onnx', 'model_fp16.onnx'), Buffer.from('fake-fp16'));
    fs.writeFileSync(path.join(outputDir, 'onnx', 'model_q4f16.onnx'), Buffer.from('old-q4f16'));

    expect(() =>
      convertOne({
        sourceDir,
        outputDir,
        spawnSyncImpl: successfulPythonSpawn(path.join(outputDir, 'onnx', 'model_q4f16.onnx')),
      })
    ).toThrow(/already exists/);
  });

  test('writes a reproducible manifest after MatMulNBits validation passes', () => {
    const sourceDir = path.join(tempRoot, 'source');
    const outputDir = path.join(tempRoot, 'output');
    const outputFile = path.join(outputDir, 'onnx', 'model_q4f16.onnx');
    fs.mkdirSync(path.join(sourceDir, 'onnx'), { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'onnx', 'model_fp16.onnx'), Buffer.from('fake-fp16'));

    const result = convertOne({
      sourceDir,
      outputDir,
      modelId: 'fixture/model',
      python: 'python',
      blockSize: 32,
      symmetric: true,
      accuracyLevel: 2,
      force: false,
      spawnSyncImpl: successfulPythonSpawn(outputFile),
    });

    const manifestPath = path.join(outputDir, Q4F16_MANIFEST_FILE);
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    expect(result).toEqual(
      expect.objectContaining({
        method: 'quantized-existing-fp16-to-q4f16',
        fp16InputSource: 'source-fp16',
        manifestPath,
      })
    );
    expect(manifest).toEqual(
      expect.objectContaining({
        modelId: 'fixture/model',
        output: expect.objectContaining({ path: outputFile, bytes: 10 }),
        externalData: expect.objectContaining({
          path: `${outputFile}.data`,
          bytes: 18,
          sha256: expect.stringMatching(/^[0-9a-f]{64}$/),
        }),
        quantization: {
          bits: DEFAULT_BITS,
          blockSize: 32,
          symmetric: true,
          quantFormat: 'QOperator',
          opTypesToQuantize: ['MatMul'],
          accuracyLevel: 2,
        },
        validation: {
          onnxChecker: 'passed',
          matMulNBitsNodeCount: 7,
          externalTensorLocations: ['model_q4f16.onnx.data'],
        },
      })
    );
  });

  test('refuses to replace the q4f16 external-data sidecar unless force is set', () => {
    const sourceDir = path.join(tempRoot, 'source');
    const outputDir = path.join(tempRoot, 'output');
    fs.mkdirSync(path.join(sourceDir, 'onnx'), { recursive: true });
    fs.mkdirSync(path.join(outputDir, 'onnx'), { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'onnx', 'model_fp16.onnx'), Buffer.from('fake-fp16'));
    fs.writeFileSync(path.join(outputDir, 'onnx', 'model_q4f16.onnx.data'), Buffer.from('old-weights'));

    expect(() =>
      convertOne({
        sourceDir,
        outputDir,
        spawnSyncImpl: successfulPythonSpawn(path.join(outputDir, 'onnx', 'model_q4f16.onnx')),
      })
    ).toThrow(/external data already exists/);
  });

  test('fails clearly when quantization does not produce the external-data sidecar', () => {
    const sourceDir = path.join(tempRoot, 'source');
    const outputDir = path.join(tempRoot, 'output');
    const outputFile = path.join(outputDir, 'onnx', 'model_q4f16.onnx');
    fs.mkdirSync(path.join(sourceDir, 'onnx'), { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'onnx', 'model_fp16.onnx'), Buffer.from('fake-fp16'));

    const spawnSyncImpl = jest.fn((command, args) => {
      const code = args[1];
      if (code.includes('MatMulNBitsQuantizer')) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, Buffer.from('embedded-q4f16'));
      }
      return { status: 0, stdout: '', stderr: '' };
    });

    expect(() =>
      convertOne({
        sourceDir,
        outputDir,
        python: 'python',
        blockSize: 32,
        symmetric: true,
        spawnSyncImpl,
      })
    ).toThrow(/did not produce the external-data sidecar/);
  });

  test('creates an fp16 q4f16 intermediate from float ONNX when source fp16 is absent', () => {
    const sourceDir = path.join(tempRoot, 'source');
    const outputDir = path.join(tempRoot, 'output');
    const outputFile = path.join(outputDir, 'onnx', 'model_q4f16.onnx');
    fs.mkdirSync(path.join(sourceDir, 'onnx'), { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'onnx', 'model.onnx'), Buffer.from('fake-float'));

    const result = convertOne({
      sourceDir,
      outputDir,
      python: 'python',
      blockSize: 32,
      symmetric: true,
      force: false,
      spawnSyncImpl: successfulPythonSpawn(outputFile),
    });

    expect(result).toEqual(
      expect.objectContaining({
        method: 'converted-float-to-fp16-then-q4f16',
        fp16InputSource: 'generated-intermediate',
        fp16Intermediate: expect.objectContaining({
          generated: true,
          source: path.join(sourceDir, 'onnx', 'model.onnx'),
          output: expect.objectContaining({
            path: path.join(outputDir, 'onnx', 'model_fp16.q4f16-intermediate.onnx'),
          }),
        }),
      })
    );
    expect(fs.existsSync(path.join(outputDir, 'onnx', 'model_fp16.q4f16-intermediate.onnx'))).toBe(true);
  });

  test('fails clearly when MatMulNBits setup is missing onnx-ir dependency', () => {
    const sourceDir = path.join(tempRoot, 'source');
    const outputDir = path.join(tempRoot, 'output');
    fs.mkdirSync(path.join(sourceDir, 'onnx'), { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'onnx', 'model_fp16.onnx'), Buffer.from('fake-fp16'));
    const spawnSyncImpl = jest.fn(() => ({
      status: 1,
      stdout: '',
      stderr: "ModuleNotFoundError: No module named 'onnx_ir'",
    }));

    expect(() =>
      convertOne({
        sourceDir,
        outputDir,
        python: 'python',
        blockSize: 32,
        symmetric: true,
        spawnSyncImpl,
      })
    ).toThrow(/onnx-ir package, imported as onnx_ir/);
  });

  test('reports ONNX checker and MatMulNBits inspection failures', () => {
    const sourceDir = path.join(tempRoot, 'source');
    const outputDir = path.join(tempRoot, 'output');
    const outputFile = path.join(outputDir, 'onnx', 'model_q4f16.onnx');
    fs.mkdirSync(path.join(sourceDir, 'onnx'), { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'onnx', 'model_fp16.onnx'), Buffer.from('fake-fp16'));

    const spawnSyncImpl = jest.fn((command, args) => {
      const code = args[1];
      if (code.includes('MatMulNBitsQuantizer')) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, Buffer.from('fake-q4f16'));
        fs.writeFileSync(`${outputFile}.data`, Buffer.from('fake-q4f16-weights'));
      }
      if (code.includes('onnx.checker.check_model')) {
        return {
          status: 1,
          stdout: '',
          stderr: 'RuntimeError: Expected at least one com.microsoft::MatMulNBits node',
        };
      }
      return { status: 0, stdout: '', stderr: '' };
    });

    expect(() =>
      convertOne({
        sourceDir,
        outputDir,
        python: 'python',
        blockSize: 32,
        symmetric: true,
        force: false,
        spawnSyncImpl,
      })
    ).toThrow(/Generated q4f16 ONNX validation failed/);
  });
});
