import home from './home.png'
import { Link } from "react-router-dom";
import '../App.css';
import {useState} from "react";
import axios from 'axios';
import RadarChart from 'react-svg-radar-chart';
export default function Model() {
  const [file, setFile] = useState();
  const [tag,setTag] = useState();
  const [imagesrc, setImagesrc] = useState();
  const [print, setPrint] = useState();
  const [result, setResult] = useState([]);
  const data = []
  function encodeFileToBase64(fileBlob) {
      const reader = new FileReader();
      reader.readAsDataURL(fileBlob);
      return new Promise((resolve)=> {
          reader.onload =() => {
              setImagesrc(reader.result);
              resolve();
          };
      });
  };
  function handleChange(e){
    setFile(e.target.files[0]);
    encodeFileToBase64(e.target.files[0]);
};
  function handleTagChange(){
    setTag('inference');
    setPrint('Now wait for the inference');
  };
  function handleSubmit(e) {
    e.preventDefault();
    if (!file) {
        console.error("파일을 업로드 해주세요");
        alert ("파일을 업로드 해주세요");
        return
    };
    const url = `http://localhost:4000/model?tag=${tag}`; 
    const formData = new FormData(); 
    // for (let i = 0 ; i < file.length ; i++) {
    //     formData.append("file", file[i]);
    // }
    formData.append("file",file);
     const config = {
            headers: {
                'content-type': 'multipart/form-data',
            }
        };
      
    axios.post(url, formData, config).then((response) => {
            console.log(response.data);
            const prediction_arr = JSON.parse(response.data.arr).slice(1, -1);
             const arr_count = [0,0,0,0,0,0,0]
            for(let i=0 ; i < prediction_arr.length; i++){
              console.log(prediction_arr[i], arr_count)
                arr_count[prediction_arr[i]]++;
            }
            let sum = 0
            for (let i=0; i< arr_count.length; i++) {
              sum = sum+arr_count[i]
            }
            for (let i = 0; i < arr_count.length; i++) {
              arr_count[i] = arr_count[i] / sum
            }
            console.log(arr_count,prediction_arr);
            setResult([arr_count]);
            setPrint('Result');
        });
        
    
    
  


};
    return (
      <main>
        <h1 className>Inference</h1>
        <p className = 'btn3'>
        <Link to="/">
          <img src = {home} alt = 'home btn'/>
        </Link>
        </p>
        <div className='inference'>
        <form onSubmit={handleSubmit} >
          <input id ="upload" required type = "file" accept="image/*" onChange={handleChange}/>
          <button type = "submit" onClick={handleTagChange}>Upload and View Inference</button>
          
        </form>
        <div className="preview">
        {imagesrc&&<img src={imagesrc} alt="preview-img"/>}
             
        {/* <h3>{print}</h3>   */}
        <ul>{
                    result.map((res)=>{
                        return <div>
                        <p>Angry: {res[0].toFixed(2)}</p>
                        <p>Disgust: {res[1].toFixed(2)}</p>
                        <p>Fear: {res[2].toFixed(2)}</p>
                        <p>Happy: {res[3].toFixed(2)}</p>
                        <p>Sad: {res[4].toFixed(2)}</p>
                        <p>Surprise: {res[5].toFixed(2)}</p>
                        <p>Neutral: {res[6].toFixed(2)}</p>
                        </div>
                    })
                }
                </ul> 
        </div>
        </div>
      </main>
    );
};