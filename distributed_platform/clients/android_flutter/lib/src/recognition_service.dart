import 'dart:convert';
import 'package:http/http.dart' as http;

class RecognitionService {
  RecognitionService({required this.baseUrl});
  final String baseUrl;
  String? _token;
  String? _deviceId;

  Future<String> login({required String username, required String password}) async {
    final uri = Uri.parse('$baseUrl/api/v1/auth/token');
    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    );
    if (resp.statusCode != 200) throw Exception('Login failed: ${resp.body}');
    _token = jsonDecode(resp.body)['access_token'] as String;
    return _token!;
  }

  Future<String> heartbeat({
    required String token,
    required String deviceName,
    required String deviceType,
    required String mode,
  }) async {
    final uri = Uri.parse('$baseUrl/api/v1/devices/heartbeat');
    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer $token'},
      body: jsonEncode({
        'device_id': _deviceId,
        'device_name': deviceName,
        'device_type': deviceType,
        'network_mode': mode,
        'status': 'online',
        'metadata': {'client': 'flutter'},
      }),
    );
    if (resp.statusCode != 200) throw Exception('Heartbeat failed: ${resp.body}');
    _deviceId = jsonDecode(resp.body)['id'] as String;
    return _deviceId!;
  }

  Future<void> sendEmbedding(List<double> embedding) async {
    final token = _token;
    final deviceId = _deviceId;
    if (token == null || deviceId == null) throw Exception('Service is not initialized.');
    final uri = Uri.parse('$baseUrl/api/v1/recognitions/match');
    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer $token'},
      body: jsonEncode({
        'device_id': deviceId,
        'embedding': embedding,
        'metadata': {'client': 'flutter'},
      }),
    );
    if (resp.statusCode != 200) throw Exception('Recognition failed: ${resp.body}');
  }
}

