import "dotenv/config";
import express, { json } from "express";
import cors from "cors";
import uploadMiddleware from "./upload.mjs";

const app = express();

const PORT = process.env.PORT || 4000;

app.use(
  cors({
    origin: "*",
  })
);
app.use(json());
app.use(express.urlencoded({ extended: false }));

app.get("/", (req, res) => res.send("success"));
app.post("/upload", uploadMiddleware.single('file'), (req, res) => {
  res.json(req.file)
})

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
