import React, { useEffect, useState } from "react";
import { SafeAreaView, ScrollView, Text, TouchableOpacity, View } from "react-native";
import { RecognitionApi, resolveRoute } from "./hybridClient";
import { WsClient } from "./wsClient";

export default function App() {
  const [events, setEvents] = useState<string[]>([]);
  const [api, setApi] = useState<RecognitionApi | null>(null);
  const [ws, setWs] = useState<WsClient | null>(null);
  const [mode, setMode] = useState<string>("initializing");

  useEffect(() => {
    let mounted = true;
    (async () => {
      const route = await resolveRoute("http://192.168.1.75:9000", "https://YOUR_PROJECT.supabase.co/functions/v1");
      const client = new RecognitionApi(route.baseUrl);
      const token = await client.login("admin", "ChangeMe123!");
      const deviceId = await client.heartbeat("android-rn-01", "android_react_native", route.mode);
      const listener = new WsClient(route.wsUrl, token, msg => {
        if (!mounted) return;
        setEvents(prev => [msg, ...prev].slice(0, 80));
      });
      listener.start();
      if (!mounted) return;
      setMode(`${route.mode} / device=${deviceId}`);
      setApi(client);
      setWs(listener);
    })();
    return () => {
      mounted = false;
      ws?.stop();
    };
  }, []);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#08121a" }}>
      <View style={{ padding: 16 }}>
        <Text style={{ color: "#d8f4ff", fontSize: 18 }}>RN Face Client ({mode})</Text>
      </View>
      <TouchableOpacity
        style={{ margin: 16, padding: 12, backgroundColor: "#1f6f8b", borderRadius: 10 }}
        onPress={async () => {
          if (!api) return;
          const fakeEmbedding = Array.from({ length: 512 }, () => 0.01);
          await api.sendEmbedding(fakeEmbedding);
        }}
      >
        <Text style={{ color: "white" }}>Send Test Embedding</Text>
      </TouchableOpacity>
      <ScrollView style={{ flex: 1, padding: 16 }}>
        {events.map((entry, idx) => (
          <Text key={`${idx}-${entry}`} style={{ color: "#9ed0e0", marginBottom: 8 }}>
            {entry}
          </Text>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

