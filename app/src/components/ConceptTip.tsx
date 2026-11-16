// ConceptTip —— 参数旁的"?"小贴士：点击弹出"术语 → 人话 → 迷宫里怎么体现"

import * as Popover from "@radix-ui/react-popover"
import { HelpCircle } from "lucide-react"

import { CONCEPTS, type ConceptKey } from "@/lib/concepts"

export function ConceptTip({ k }: { k: ConceptKey }) {
  const c = CONCEPTS[k]
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          type="button"
          aria-label={`${c.term} 是什么？`}
          className="inline-flex align-middle text-slate-400 transition-colors hover:text-brand"
        >
          <HelpCircle className="h-3.5 w-3.5" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="right"
          align="start"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 w-72 space-y-1.5 rounded-md border border-slate-200 bg-white p-3 text-xs text-slate-700 shadow-lg"
        >
          <div className="text-sm font-semibold text-ink">{c.term}</div>
          <div className="rounded bg-sky-50 px-2 py-1 text-sky-900">💡 人话：{c.plain}</div>
          <p className="leading-relaxed">{c.detail}</p>
          {"formula" in c && c.formula && (
            <div className="rounded bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-800">{c.formula}</div>
          )}
          <Popover.Arrow className="fill-white" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  )
}
