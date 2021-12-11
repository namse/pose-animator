import { createServer } from "http";
import { WebSocketServer } from "ws";
import { getLandmarks, putLandmark, removeLandmark } from "./landmarks";
import { serveFile } from "./serveFile";

const server = createServer((req, res) => {
  if (req.method === "GET") {
    return serveFile(req, res);
  }
});
const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  const id =
    Math.random().toString(36).substring(2, 15) +
    Math.random().toString(36).substring(2, 15);
  ws.send(
    JSON.stringify({
      type: "setId",
      id,
    })
  );

  ws.on("message", (data) => {
    const landmark = JSON.parse(data.toString());
    putLandmark(id, landmark);
    broadcastLandmarks();
  });

  ws.on("close", () => {
    console.log("close");
    removeLandmark(id);
    broadcastLandmarks();
  });
});

function broadcastLandmarks() {
  const landmarks = getLandmarks();
  wss.clients.forEach((client) => {
    client.send(
      JSON.stringify({
        type: "landmarks",
        landmarks,
      })
    );
  });
}

server.listen(8080, "0.0.0.0");
