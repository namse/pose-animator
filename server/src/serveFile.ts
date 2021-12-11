import { IncomingMessage, ServerResponse } from "http";
import path from "path";
import fs from "fs/promises";

export async function serveFile(req: IncomingMessage, res: ServerResponse) {
  if (!req.url) {
    return;
  }
  const filePath = path.join(
    __dirname,
    "../../pose-animator/dist",
    req.url === "/" ? "index.html" : req.url
  );

  try {
    const file = await fs.readFile(filePath);
    res.writeHead(200);
    res.end(file);
  } catch (error) {
    if (error === "ENOENT") {
      res.statusCode = 404;
      res.end("File not found");
    }
  }
}
