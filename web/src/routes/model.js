import home from './home.png'
import { Link } from "react-router-dom";
import '../App.css';
import {useState} from "react";
import axios from 'axios';
export default function Model() {
  const [file, setFile] = useState();
  const [tag,setTag] = useState();
  const [imagesrc, setImagesrc] = useState();
  const [print, setPrint] = useState()
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
    const url = `http://localhost:4000/upload?tag=${tag}`;
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
        <h3>{print}</h3>
        </div>
        </div>
      </main>
    );
};