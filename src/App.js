import React,{useState} from "react";
import axios from "axios";

function App() {
    const [file, setFile] = useState();
    function handleChange(e) {
        setFile(e.target.files[0]);
    };
    function handleSubmit(e) {
        e.preventDefault();
        const url = 'http://localhost:4000/upload';
        const formData = new FormData();
            formData.append('file',file);
            // formData.append('fileName',file.name);
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
            <form onSubmit={handleSubmit}>
                <h1>File Uplaod</h1>
                <input type = "file" accept="image/*" onChange={handleChange}/>
                <button type = "submit">Upload</button>
            </form>
        </div>
    );
};
export default App