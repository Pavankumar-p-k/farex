import 'dart:convert';
import 'package:http/http.dart' as http;

class HybridRoute {
  HybridRoute({required this.mode, required this.baseUrl, required this.wsUrl});
  final String mode;
  final String baseUrl;
  final String wsUrl;
}

class HybridEndpointResolver {
  HybridEndpointResolver({required this.localBase, required this.cloudBase});
  final String localBase;
  final String cloudBase;

  Future<HybridRoute> resolve() async {
    try {
      final health = Uri.parse('$localBase/api/v1/health');
      final resp = await http.get(health).timeout(const Duration(milliseconds: 1200));
      if (resp.statusCode == 200) {
        return HybridRoute(mode: 'local', baseUrl: localBase, wsUrl: 'ws://${Uri.parse(localBase).host}:9000/ws/events');
      }
    } catch (_) {
      // Fallback to cloud
    }
    return HybridRoute(
      mode: 'cloud',
      baseUrl: cloudBase,
      wsUrl: 'wss://YOUR_PROJECT.supabase.co/realtime/v1/websocket',
    );
  }
}

