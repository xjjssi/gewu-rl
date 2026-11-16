// ConceptGuide —— "RL 与人类决策"速查面板
//
// 把 RL 的核心概念与"迷宫里的样子""生活中人的做法"三列对照，
// 并引用《强化学习》（Sutton & Barto）第 1 章的几个直观例子，
// 说明 RL 的试错学习与人做决策是同一种思维。

import { X } from "lucide-react"

const ROWS: { term: string; maze: string; human: string }[] = [
  { term: "状态 s", maze: "智能体所在的格子", human: "你现在在哪、处境如何" },
  { term: "动作 a", maze: "上 / 下 / 左 / 右", human: "眼下可以做的选择" },
  { term: "奖励 r", maze: "每步 −0.01，撞墙扣分，终点 +1", human: "即时的甜头或苦头" },
  { term: "Q 值 Q(s,a)", maze: "每格四个方向各一个分数", human: "对每个选择的“印象分”" },
  { term: "策略 π", maze: "迷宫上的箭头", human: "做事的习惯 / 套路" },
  { term: "探索", maze: "以概率 ε 随机走一步", human: "去一家没吃过的新餐馆" },
  { term: "寻优（利用）", maze: "选 Q 值最大的方向", human: "去常去、口碑最好的那家" },
  { term: "TD 更新", maze: "结果比预期好 → Q 上调", human: "吃一堑，长一智" },
  { term: "折现因子 γ", maze: "终点奖励往回传时逐格打折", human: "远见：先苦后甜还是只顾眼前" },
  { term: "价值热力图", maze: "离终点越近颜色越蓝", human: "离家越近越安心" },
]

const BOOK_EXAMPLES: { title: string; text: string; link: string }[] = [
  {
    title: "扫地机器人要不要进新房间？",
    text: "电量够就去新房间找垃圾（探索），电量低就按以往经验回去充电（寻优）。",
    link: "对应迷宫里每一步的“探索 / 寻优”与 ε-greedy。",
  },
  {
    title: "小瞪羚出生后学跑步",
    text: "刚出生时跌跌撞撞，半小时后就能奔跑——全靠与环境互动、试错改进。",
    link: "对应早期回合乱走、后期走出最短路的学习曲线。",
  },
  {
    title: "棋手下棋",
    text: "既会预判对手的反应，也凭“这步棋好不好”的直觉判断。",
    link: "对应 Q 值：对“在这个局面下走这步”的价值估计。",
  },
]

export function ConceptGuide({ onClose }: { onClose: () => void }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-start gap-2">
        <div>
          <h2 className="text-base font-semibold text-ink">📖 RL 与人类决策：一张对照表看懂强化学习</h2>
          <p className="mt-0.5 text-xs text-slate-500">
            强化学习模仿的正是人“试一试 → 看结果 → 调整做法”的思维。迷宫里的状态、动作、奖励和价值都能直接看见，所以适合用来讲基础理论。
          </p>
        </div>
        <button type="button" onClick={onClose} aria-label="关闭"
          className="ml-auto rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500">
              <th className="py-1.5 pr-2 font-medium">RL 概念</th>
              <th className="py-1.5 pr-2 font-medium">在迷宫里</th>
              <th className="py-1.5 font-medium">换成人</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.term} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5 pr-2 font-semibold text-slate-900">{r.term}</td>
                <td className="py-1.5 pr-2 text-slate-700">{r.maze}</td>
                <td className="py-1.5 text-slate-700">{r.human}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="space-y-2">
          <div className="text-xs font-semibold text-slate-900">《强化学习》（Sutton &amp; Barto）第 1 章中的直观例子</div>
          {BOOK_EXAMPLES.map((e) => (
            <div key={e.title} className="rounded border border-slate-200 bg-slate-50 p-2 text-xs">
              <div className="font-semibold text-slate-900">{e.title}</div>
              <p className="mt-0.5 text-slate-700">{e.text}</p>
              <p className="mt-0.5 text-brand">→ {e.link}</p>
            </div>
          ))}
          <p className="text-[11px] text-slate-500">
            想看智能体每一步是怎么想的？训练 Q-Learning 后，在“训练实时视图”里暂停回放，查看“决策解说”。
          </p>
        </div>
      </div>
    </div>
  )
}
