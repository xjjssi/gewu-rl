import { Routes, Route } from 'react-router'
import Home from './pages/Home'

// 单页应用：任意路径都渲染主界面，便于部署到 GitHub Pages / Gitee Pages 等子路径（如 /gewu-rl/）
export default function App() {
  return (
    <Routes>
      <Route path="*" element={<Home />} />
    </Routes>
  )
}
