import { HashRouter, Routes, Route  } from "react-router"
import Home from "./Home"
import Viewer from "./Viewer"


function App() {
  return <HashRouter>
    <Routes>
      <Route path="/viewer/:jsonFile" element={<Viewer />} />
      <Route path="/" element={<Home />} />
    </Routes>
  </HashRouter>
}

export default App
