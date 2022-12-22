import sadpic from './sad.jpg';
import angrypic from './angry.jpg';
import disgustpic from './disgust.jpg';
import neutralpic from './neautral.jpg';
import fearpic from './fear.jpg';
import happypic from './happy.jpg';
import surprisepic from './surprise.jpg';
import home from './home.png'
import { Link } from "react-router-dom";
import '../App.css'
export default function Model() {
    return (
      <main>
        <h1 className>Inference</h1>
        <p className = 'btn3'>
        <Link to="/">
          <img src = {home} alt = 'home btn'/>
        </Link>
        </p>
        <p className='view'>
          <img src = {angrypic} alt ='angry'/>
        </p>
        <p className='view'>
          <img src = {disgustpic} alt='disgust'/>
        </p>
        <p className='view'>
          <img src ={fearpic} alt='fear'/>
        </p>
        <p className='view'>
          <img src = {happypic} alt='happy'/>
        </p>
        <p className='view'>
          <img src = {sadpic} alt = 'sad'/>
        </p>
        <p className='view'>
          <img src = {surprisepic} alt='surprise'/>
        </p>
        <p className='view'>
          <img src = {neutralpic} alt='neutral'/>
        </p>
      </main>
    );
};