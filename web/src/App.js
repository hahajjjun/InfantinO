import { Link } from "react-router-dom";
import React from 'react';
import "./App.css";
import uploadimg from './upload.png';
import modelimg from './model.png';
export default function App() {
  return (
    <div>
      <h1 className = 'title'>InfantinO</h1>
      <div className ='intro'>
        <p>InfantinO is an Online Learning Framework for classifying and learning features of Infants' facial expressions.</p>
        <p>We aim to continually update our ML model, using Online Learning, by training just one photo at a time.</p>
        <p>The picture below shows how our model works.</p>
        <p></p>
      </div>
      <p>
      </p>
      <div className='btn1'>
        <Link to="/upload">
          <p><img src ={uploadimg} alt='imgbtn'/></p>
          <p className='link'>Upload Image</p>
        </Link>
      </div>
      <div className='btn2'>
        <Link to="/model" >
          <p><img src = {modelimg} alt='imgbtn'/></p>
          <p className ='link'>View Model</p>
        </Link>
      </div>
    </div>
  );
};