import 'package:flutter/material.dart';
import 'src/hybrid_endpoint.dart';
import 'src/recognition_service.dart';
import 'src/ws_service.dart';

void main() {
  runApp(const FaceClientApp());
}

class FaceClientApp extends StatefulWidget {
  const FaceClientApp({super.key});

  @override
  State<FaceClientApp> createState() => _FaceClientAppState();
}

class _FaceClientAppState extends State<FaceClientApp> {
  final List<String> _events = <String>[];
  final HybridEndpointResolver _resolver = HybridEndpointResolver(
    localBase: 'http://192.168.1.75:9000',
    cloudBase: 'https://YOUR_PROJECT.supabase.co/functions/v1',
  );

  RecognitionService? _api;
  WsService? _ws;
  String _mode = 'resolving';

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    final route = await _resolver.resolve();
    final api = RecognitionService(baseUrl: route.baseUrl);
    final token = await api.login(username: 'admin', password: 'ChangeMe123!');
    final deviceId = await api.heartbeat(
      token: token,
      deviceName: 'android-flutter-01',
      deviceType: 'android_flutter',
      mode: route.mode,
    );

    final ws = WsService(
      wsUrl: route.wsUrl,
      token: token,
      onEvent: (msg) {
        if (!mounted) return;
        setState(() {
          _events.insert(0, msg);
          if (_events.length > 60) _events.removeLast();
        });
      },
    );
    ws.start();

    if (!mounted) return;
    setState(() {
      _mode = '${route.mode} / device=$deviceId';
      _api = api;
      _ws = ws;
    });
  }

  @override
  void dispose() {
    _ws?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Face Client ($_mode)')),
        body: ListView.builder(
          itemCount: _events.length,
          itemBuilder: (_, index) => ListTile(title: Text(_events[index])),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: () async {
            // Real app: replace with ArcFace embedding from camera frame.
            final fakeEmbedding = List<double>.filled(512, 0.01);
            final api = _api;
            if (api == null) return;
            await api.sendEmbedding(fakeEmbedding);
          },
          child: const Icon(Icons.play_arrow),
        ),
      ),
    );
  }
}

