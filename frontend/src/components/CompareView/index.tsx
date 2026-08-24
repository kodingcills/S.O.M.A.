// PLACEHOLDER — Task 2.1 scaffold only.
// Real comparison view (dual MJPEG streams + stats bar) lands in Task 2.5.
import { CANVAS_BG, TEXT_SECONDARY, PANEL_BORDER } from '../../utils/colors'

export function CompareView() {
  return (
    <div style={{
      width: '100%', height: '100%', background: CANVAS_BG,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      borderBottom: `1px solid ${PANEL_BORDER}`,
      color: TEXT_SECONDARY,
      fontFamily: "'JetBrains Mono', ui-monospace, Menlo, monospace",
      fontSize: 12, letterSpacing: '0.15em',
    }}>
      COMPARE VIEW
    </div>
  )
}
