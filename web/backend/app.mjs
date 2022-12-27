import express, { json } from "express";
import cors from "cors";
import uploadMiddleware from "./upload.mjs";
import axios from "axios"
// import { createRequire } from 'module';
// const require = createRequire(import.meta.url);
// const request2 = require('request');
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

app.post("/model", uploadMiddleware.single('file'), (req, res,error) => {
  const img_s3_path = req.file.location
  const url = 'https://endpoint-ybigta-4.eastus2.inference.ml.azure.com/score'
   const options = { 
    "headers":{
        
        
        "content-type": 'application/json',
        "Authorization": 'Bearer orstLDZNpez7wcjxjUSeam0GmyA6TqDe',
        "azureml-model-deployment" : 'defaultname-4'
    },

   
    }
    axios.post( url, {data:img_s3_path} ,options).then(function (response) {
      console.log(response.data);
    
      res.json({url: req.file.location, arr: response.data})
      
  }).catch(console.error);
});




  


app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

