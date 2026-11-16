# 参与贡献

欢迎老师、同学和开发者参与格物RL。贡献方式按门槛从低到高：

1. **投稿教学案例**：用页面右上角"分享实验任务"生成链接，按 Issue 模板"教学案例投稿"提交，好的案例会收录进 `docs/教学/`。
2. **反馈问题**：按"问题反馈"模板提交，附上实验任务链接可一键复现。
3. **改进代码或文档**：Fork → 新建分支 → 提交 PR（模板中有自检清单）。

## 开发环境

```bash
pip install -r requirements.txt          # Python 3.10+（验证版本 3.13）
cd app && npm ci                          # Node.js 20.19+ 或 22.12+
python -m pytest                          # 根目录运行全部测试
cd app && npm run build                   # 类型检查 + 生产构建
```

## 约定

- **算法同步**：训练算法有两份实现——后端 `gridworld_edu`（Python）与浏览器引擎 `app/src/lib/localEngine.ts`
  （在线体验版）。改动其一时请同步另一份；`tests/test_browser_engine.py` 会在 4 个预设 × 10 个种子上检查两者一致。
- **预设迷宫**：以 `gridworld_edu/envs/presets.py` 为准，`app/src/lib/presets.ts` 由其生成，测试会校验逐格一致。
- **正式实验数据**：`backend/results/report/` 为技术报告引用的正式结果，请勿手工修改；重跑见 `backend/report_experiments.py`。
- 代码注释与文档使用中文，面向教学读者，公式与教材记号保持一致。

## 许可

提交的贡献以 MIT 许可证发布（见 LICENSE）。请勿提交无权再分发的第三方内容、个人信息或访问凭据。
