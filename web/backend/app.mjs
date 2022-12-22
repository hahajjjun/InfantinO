// import "dotenv/config";
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
  res.json({url: req.file.location});
})

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});


//  try {
//   console.log('req.file.location: ', req.file.location) //single : req.file, array : req.files 
//   await insertProfileImgToDb(myQurey.insertProfileImg, req.file.location)
//   res.send(req,file.location) //client에게 s3 이미지 경로 반환
// } catch (error) {
//   console.log('Enter error: ', error);   