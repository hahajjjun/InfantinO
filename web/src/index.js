import ReactDOM from 'react-dom/client';
import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";
import App from "./App";
import Model from "./routes/model";
import Upload from "./routes/upload";

const rootElement = ReactDOM.createRoot(document.getElementById("root"));
rootElement.render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="model" element={<Model />} />
      <Route path="upload" element={<Upload />} />
    </Routes>
  </BrowserRouter>
);