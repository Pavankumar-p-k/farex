export class WsClient {
  private socket: WebSocket | null = null;

  constructor(
    private readonly wsUrl: string,
    private readonly token: string,
    private readonly onMessage: (msg: string) => void,
  ) {}

  start(): void {
    this.socket = new WebSocket(`${this.wsUrl}?token=${this.token}`);
    this.socket.onmessage = event => this.onMessage(String(event.data));
    this.socket.onerror = err => this.onMessage(`ws_error: ${JSON.stringify(err)}`);
    this.socket.onclose = () => this.onMessage("ws_closed");
  }

  stop(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}

