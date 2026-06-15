import type { Message } from '../shared/message-types';

/**
 * Chrome can throw synchronously after an extension reload invalidates the
 * content-script context. Promise .catch() only handles async rejections.
 */
export function sendRuntimeMessageBestEffort(message: Message): void {
  try {
    void chrome.runtime.sendMessage(message).catch(() => undefined);
  } catch {
    // Best effort: stale content scripts cannot talk to the new extension
    // context until the page is refreshed.
  }
}
