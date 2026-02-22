# Android Client Implementation Notes

## Flutter Client
- Location: `clients/android_flutter`
- Current scaffold includes:
  - Hybrid route resolution (local first, cloud fallback)
  - REST login + heartbeat + embedding submit
  - WebSocket live event subscription
- To complete production app:
  1. Add CameraX/camera plugin stream.
  2. Integrate ArcFace-compatible on-device embedding model (ONNX/TFLite).
  3. Replace placeholder embedding in `main.dart`.

## React Native Client
- Location: `clients/android_react_native`
- Current scaffold includes:
  - Same hybrid route logic as Flutter
  - REST + WebSocket service classes
  - Basic UI event feed
- To complete production app:
  1. Add camera module (`react-native-vision-camera`).
  2. Add native ONNX/TFLite bridge for ArcFace embeddings.
  3. Send normalized 512-d embedding only.

## Shared Contract
- Both clients call same backend endpoints:
  - `/api/v1/auth/token`
  - `/api/v1/devices/heartbeat`
  - `/api/v1/recognitions/match`
  - `WS /ws/events`

