import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Defs, Rect, Mask, Ellipse, Text as SvgText } from 'react-native-svg';

// ─── Oval geometry (must match the cutout below) ─────────────────────────────
// cx="50%", cy="45%" in SVG space.  Arrow offset from oval centre in pixels.
const OVAL_RX = 120;
const OVAL_RY = 160;

export interface FaceMaskOverlayProps {
  /**
   * Centering status driven by the worklet thread via onCentering callback.
   *   null  = no face detected (waiting)
   *   false = face visible but off-centre
   *   true  = face is centred — show green confirmation
   */
  centered: boolean | null;
  /**
   * Normalised face-centre position (0-1) from the worklet.
   * Used to derive which directional arrows to show.
   * null when no face is detected.
   */
  facePos: { cx: number; cy: number } | null;
}

/**
 * Derive up to two directional arrow glyphs from the face centre.
 * The face centre is in normalised 0-1 space; target oval centre is (0.50, 0.45).
 * Dead-band of ±0.05 suppresses arrows for near-centred wobble.
 */
function getArrows(pos: { cx: number; cy: number } | null): string {
  if (!pos) return '';
  const arrows: string[] = [];
  // Horizontal
  if (pos.cx < 0.45)  arrows.push('→');
  else if (pos.cx > 0.55) arrows.push('←');
  // Vertical
  if (pos.cy < 0.40)  arrows.push('↓');
  else if (pos.cy > 0.50) arrows.push('↑');
  return arrows.join('  ');
}

export function FaceMaskOverlay({ centered, facePos }: FaceMaskOverlayProps) {
  // ── Oval stroke style based on centering status ──────────────────────────
  const strokeColor =
    centered === true  ? 'rgba(34,197,94,0.95)'  :  // green  — centred
    centered === false ? 'rgba(239,68,68,0.85)'   :  // red    — off-centre
                        'rgba(255,255,255,0.5)';     // white  — no face

  const strokeWidth   = centered === true ? 3 : 2;
  const strokeDash    = centered === true ? undefined : '8,8';

  // ── Directional arrows / checkmark ───────────────────────────────────────
  const arrowText = centered === false ? getArrows(facePos) : '';
  const showCheck = centered === true;

  return (
    // pointerEvents="none" is critical so the overlay doesn't block UI touches.
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <Svg height="100%" width="100%">
        <Defs>
          <Mask id="face-hole">
            {/* White = visible, black = punched-out hole */}
            <Rect x="0" y="0" width="100%" height="100%" fill="white" />
            <Ellipse cx="50%" cy="45%" rx={OVAL_RX} ry={OVAL_RY} fill="black" />
          </Mask>
        </Defs>

        {/* Dark overlay with the face oval cut out */}
        <Rect
          x="0" y="0" width="100%" height="100%"
          fill="rgba(0,0,0,0.6)"
          mask="url(#face-hole)"
        />

        {/* Oval border — colour and style reflect centering status */}
        <Ellipse
          cx="50%" cy="45%"
          rx={OVAL_RX} ry={OVAL_RY}
          fill="transparent"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={strokeDash}
        />

        {/* ✓ checkmark when centred */}
        {showCheck && (
          <SvgText
            x="50%" y="45%"
            textAnchor="middle"
            alignmentBaseline="central"
            fontSize={52}
            fill="rgba(34,197,94,0.95)"
          >
            ✓
          </SvgText>
        )}

        {/* Directional arrows when off-centre */}
        {!showCheck && arrowText !== '' && (
          <SvgText
            x="50%" y="45%"
            textAnchor="middle"
            alignmentBaseline="central"
            fontSize={40}
            fill="rgba(255,255,255,0.95)"
          >
            {arrowText}
          </SvgText>
        )}
      </Svg>
    </View>
  );
}
