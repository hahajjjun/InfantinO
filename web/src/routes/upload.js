import React,{useState} from "react";
import axios from "axios";
import './subpage.css';
import { Link } from "react-router-dom";
import home from './home.png';
function Upload() {
    const [file, setFile] = useState();
    const [tag,setTag] = useState();
    const [imagesrc, setImagesrc] = useState('');
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
    function labelZero() {
        let label='angry';
        setTag(label);
        setPrint('The file has been labeled and sent to the server.');
        console.log(tag);
};
    function labelOne() {
        let label='disgust';
        setTag(label);
        setPrint('The file has been labeled and sent to the server.');
        console.log(tag);
};      
    function labelTwo() {
        let label='fear';
        setTag(label);
        setPrint('The file has been labeled and sent to the server.');
        console.log(tag);
};
    function labelThree() {
        let label='happy';
        setTag(label);
        setPrint('The file has been labeled and sent to the server.');
        console.log(tag);
};
    function labelFour() {
        let label='sad';
        setTag(label);
        setPrint('The file has been labeled and sent to the server.');
        console.log(tag);
};
    function labelFive() {
        let label='surprise';
        setTag(label);
        setPrint('The file has been labeled and sent to the server.');
        console.log(tag);
};
    function labelSix() {
        let label='neutral';
        setTag(label);
        setPrint('The file has been labeled and sent to the server')
        console.log(tag);
};
    function handleChange(e){
        setFile(e.target.files[0]);
        encodeFileToBase64(e.target.files[0]);
        setPrint('The file hasn\'t been sent to the server yet.')
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
        <div className = "App">
            <p className = 'btn3'>
                <Link to="/">
                    <img src = {home} alt = 'home btn'/>
                </Link>
            </p>
            <h1 className="title1">File Upload</h1>
            <div className="imguploadform">
            <form onSubmit={handleSubmit} >
                <input id ="upload" required type = "file" accept="image/*" onChange={handleChange}/>
                <div id='wrapper' className='labeling'>
                    <p className='label1'>
                        <button type='submit' className='labelbtn' onClick={labelZero}>1</button> Angry
                    </p>
                    <p className='label2'>
                        <button type='submit' className='labelbtn' onClick={labelOne}>2</button> Disgust
                    </p>
                    <p className="label3">
                        <button type='submit' className='labelbtn' onClick={labelTwo}>3</button> Fear
                    </p>
                    <p className="label4">
                        <button type='submit' className="labelbtn" onClick={labelThree}>4</button> Happy
                    </p>
                    <p className="label5">
                        <button type='submit' className="labelbtn" onClick={labelFour}>5</button> Sad
                    </p>
                    <p className="label6">
                        <button type='submit' className="labelbtn" onClick={labelFive}>6</button> Surprise
                    </p>
                    <p className="label7">
                        <button type='submit' className="labelbtn" onClick={labelSix}>7</button> Neutral
                    </p>
                </div>
            </form>
            </div>
            <div className ="preview">
                <h2>Current Image</h2>
                {imagesrc&&<img src={imagesrc} alt="preview-img"/>}
                <h3>{print}</h3>
            </div>
        </div>
    );
};
export default Upload;