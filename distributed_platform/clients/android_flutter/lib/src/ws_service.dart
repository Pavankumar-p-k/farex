import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class WsService {
  WsService({
    required this.wsUrl,
    required this.token,
    required this.onEvent,
  });

  final String wsUrl;
  final String token;
  final void Function(String message) onEvent;
  WebSocketChannel? _channel;

  void start() {
    final uri = Uri.parse('$wsUrl?token=$token');
    _channel = WebSocketChannel.connect(uri);
    _channel!.stream.listen(
      (data) => onEvent(data.toString()),
      onError: (err) => onEvent('ws_error: $err'),
      onDone: () => onEvent('ws_done'),
    );
  }

  void dispose() {
    _channel?.sink.close();
    _channel = null;
  }
}

