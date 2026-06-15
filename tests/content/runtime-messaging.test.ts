import { sendRuntimeMessageBestEffort } from '../../src/content/runtime-messaging';

describe('sendRuntimeMessageBestEffort', () => {
  const message = {
    type: 'SUPPORTED_PAGE_ACTIVITY',
    payload: { visible: true },
  } as const;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('swallows synchronous extension-context failures', () => {
    (chrome.runtime.sendMessage as jest.Mock).mockImplementationOnce(() => {
      throw new Error('Extension context invalidated.');
    });

    expect(() => sendRuntimeMessageBestEffort(message)).not.toThrow();
  });

  it('swallows asynchronous send failures', async () => {
    (chrome.runtime.sendMessage as jest.Mock).mockRejectedValueOnce(
      new Error('Receiving end does not exist.'),
    );

    sendRuntimeMessageBestEffort(message);
    await Promise.resolve();

    expect(chrome.runtime.sendMessage).toHaveBeenCalledWith(message);
  });
});
