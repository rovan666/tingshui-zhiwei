import { test, expect } from '@playwright/test'

const URL = 'http://127.0.0.1:7100/'

test('desktop dashboard uses real data and all primary controls work', async ({ page }) => {
  const consoleErrors: string[] = []
  const pageErrors: string[] = []
  const failedResponses: string[] = []
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()) })
  page.on('pageerror', (error) => pageErrors.push(error.message))
  page.on('response', (response) => { if (response.status() >= 400) failedResponses.push(`${response.status()} ${response.url()}`) })

  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto(URL, { waitUntil: 'networkidle' })
  await expect(page.getByRole('heading', { name: '事件告警记录' })).toBeVisible()
  await expect(page.locator('tbody tr')).toHaveCount(3)
  await expect(page.locator('.simulation-badge')).toHaveText('SIMULATED · 仿真数据')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)

  const alarmChecks = [
    ['A-01', '60.9', '船舶噪声'],
    ['A-02', '119.8', '人员拍水'],
    ['A-03', '250.2', '鱼类活动'],
  ] as const
  for (const [id, bearing, label] of alarmChecks) {
    await page.getByRole('button', { name: `查看 ${id} 告警`, exact: true }).click()
    await expect(page.locator('.bearing-panel')).toContainText(bearing)
    await expect(page.locator('.classification-result')).toContainText(label)
    await expect(page.locator('.event-tag')).toHaveText(id)
  }

  for (const speed of ['2×', '4×', '1×']) {
    await page.getByRole('button', { name: speed, exact: true }).click()
    await expect(page.getByRole('button', { name: speed, exact: true })).toHaveAttribute('aria-pressed', 'true')
  }
  for (const channel of ['CH1', 'CH2', 'CH3', 'CH4']) {
    const button = page.getByRole('button', { name: channel, exact: true })
    await button.click()
    await expect(button).toHaveAttribute('aria-pressed', 'false')
    await button.click()
    await expect(button).toHaveAttribute('aria-pressed', 'true')
  }

  await page.getByRole('button', { name: '开始回放', exact: true }).click()
  await expect(page.getByRole('button', { name: '暂停回放', exact: true })).toBeVisible()
  await page.waitForTimeout(700)
  await page.getByRole('button', { name: '暂停回放', exact: true }).click()
  const pausedTime = await page.locator('.time-readout').innerText()
  expect(pausedTime).not.toContain('0.00 /')
  await page.waitForTimeout(250)
  await expect(page.locator('.time-readout')).toHaveText(pausedTime)

  const slider = page.getByRole('slider', { name: '回放时间' })
  await slider.focus()
  await page.keyboard.press('End')
  await expect(page.locator('.time-readout')).toContainText('8.00')
  await page.keyboard.press('Home')
  await expect(page.locator('.time-readout')).toContainText('0.00')

  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('link', { name: '导出数据' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('tingshui-demo-data.json')

  await page.getByRole('button', { name: '查看 A-01 告警', exact: true }).click()
  await page.getByRole('button', { name: '回到起点' }).click()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: '../figures/day4/workbench.png', fullPage: true })
  const metrics = await page.evaluate(() => {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
    const canvas = document.querySelector('canvas')!
    const pixels = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data
    return {
      viewport: [innerWidth, innerHeight],
      page: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
      resourceCount: resources.length,
      transferredBytes: resources.reduce((sum, item) => sum + item.transferSize, 0),
      canvasNonEmpty: pixels.some((value) => value !== 0),
    }
  })
  console.log(`UI_METRICS ${JSON.stringify(metrics)}`)
  expect(metrics.canvasNonEmpty).toBe(true)
  expect(consoleErrors).toEqual([])
  expect(pageErrors).toEqual([])
  expect(failedResponses).toEqual([])
})

test('mobile layout stays within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await page.goto(URL, { waitUntil: 'networkidle' })
  await expect(page.getByRole('heading', { name: '事件告警记录' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: '../figures/day4/workbench-mobile.png', fullPage: true })
})

test('data failure has a working retry state', async ({ page }) => {
  await page.route('**/demo-data.json', (route) => route.abort())
  await page.goto(URL)
  await expect(page.getByRole('heading', { name: '演示数据读取失败' })).toBeVisible()
  await page.unroute('**/demo-data.json')
  await page.getByRole('button', { name: '重新读取' }).click()
  await expect(page.getByRole('heading', { name: '事件告警记录' })).toBeVisible()
})
