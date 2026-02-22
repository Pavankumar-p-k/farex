export type RouteMode = "local" | "cloud";

export interface RouteDecision {
  mode: RouteMode;
  baseUrl: string;
  wsUrl: string;
}

export async function resolveRoute(localBase: string, cloudBase: string): Promise<RouteDecision> {
  try {
    const resp = await fetch(`${localBase}/api/v1/health`, { method: "GET" });
    if (resp.ok) {
      const localHost = new URL(localBase).host;
      return {
        mode: "local",
        baseUrl: localBase,
        wsUrl: `ws://${localHost}/ws/events`,
      };
    }
  } catch (_e) {
    // ignore and fallback
  }
  return {
    mode: "cloud",
    baseUrl: cloudBase,
    wsUrl: "wss://YOUR_PROJECT.supabase.co/realtime/v1/websocket",
  };
}

export class RecognitionApi {
  constructor(private readonly baseUrl: string) {}
  token: string | null = null;
  deviceId: string | null = null;

  async login(username: string, password: string): Promise<string> {
    const resp = await fetch(`${this.baseUrl}/api/v1/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!resp.ok) {
      throw new Error(`Login failed: ${resp.status}`);
    }
    const body = await resp.json();
    this.token = body.access_token;
    return this.token!;
  }

  async heartbeat(deviceName: string, deviceType: string, mode: RouteMode): Promise<string> {
    if (!this.token) throw new Error("Missing token");
    const resp = await fetch(`${this.baseUrl}/api/v1/devices/heartbeat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.token}`,
      },
      body: JSON.stringify({
        device_id: this.deviceId,
        device_name: deviceName,
        device_type: deviceType,
        network_mode: mode,
        status: "online",
        metadata: { client: "react_native" },
      }),
    });
    if (!resp.ok) throw new Error(`Heartbeat failed: ${resp.status}`);
    const body = await resp.json();
    this.deviceId = body.id;
    return this.deviceId!;
  }

  async sendEmbedding(embedding: number[]): Promise<void> {
    if (!this.token || !this.deviceId) throw new Error("Client not initialized");
    const resp = await fetch(`${this.baseUrl}/api/v1/recognitions/match`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.token}`,
      },
      body: JSON.stringify({
        device_id: this.deviceId,
        embedding,
        metadata: { client: "react_native" },
      }),
    });
    if (!resp.ok) throw new Error(`Recognition failed: ${resp.status}`);
  }
}

