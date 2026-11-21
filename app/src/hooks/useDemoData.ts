import { useCallback, useEffect, useState } from 'react'
import type { DemoData } from '@/types/demo'

export function useDemoData() {
  const [data, setData] = useState<DemoData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)

  const retry = useCallback(() => {
    setError(null)
    setAttempt((value) => value + 1)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${import.meta.env.BASE_URL}demo-data.json`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`数据请求失败：HTTP ${response.status}`)
        return response.json() as Promise<DemoData>
      })
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setError(reason instanceof Error ? reason.message : '无法读取演示数据')
      })
    return () => controller.abort()
  }, [attempt])

  return { data, error, retry }
}
