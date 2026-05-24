import React from 'react';
import { View, StyleSheet, useWindowDimensions } from 'react-native';
import Svg, { Path, Ellipse, Text as SvgText } from 'react-native-svg';

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
  const { width, height } = useWindowDimensions();

  // Oval centre in pixels (must match cx="50%" cy="45%" used for the border ellipse)
  const cx = width  * 0.5;
  const cy = height * 0.45;

  // ── Oval stroke style based on centering status ──────────────────────────
  const strokeColor =
    centered === true  ? 'rgba(34,197,94,0.95)'  :  // green  — centred
    centered === false ? 'rgba(239,68,68,0.85)'   :  // red    — off-centre
                        'rgba(255,255,255,0.7)';     // white  — no face / waiting

  const strokeWidth = centered === true ? 3 : 2;
  const strokeDash  = centered === true ? undefined : '8,6';

  // ── Directional arrows / checkmark ───────────────────────────────────────
  const arrowText = centered === false ? getArrows(facePos) : '';
  const showCheck = centered === true;

  // ── Compound path: full-screen rect  ∪  ellipse, filled with evenodd rule ─
  //
  // fillRule="evenodd" makes the overlapping ellipse region count as "outside"
  // (even winding) → transparent → camera shows through.  The surrounding rect
  // area counts as "inside" (odd winding) → gets the dark fill.
  //
  // This replaces the SVG <Mask> approach which fails to apply on the very first
  // render on Android (react-native-svg bug), causing the oval to appear as a
  // solid white shape until the component re-renders.
  //
  // Ellipse arc in SVG path notation:
  //   M (cx-rx) cy  — start at left edge of ellipse
  //   A rx ry 0 1 0 (cx+rx) cy  — large arc to right edge (top half)
  //   A rx ry 0 1 0 (cx-rx) cy  — large arc back to left edge (bottom half)
  //   Z
  const overlayPath = [
    `M 0 0 H ${width} V ${height} H 0 Z`,
    `M ${cx - OVAL_RX} ${cy}`,
    `A ${OVAL_RX} ${OVAL_RY} 0 1 0 ${cx + OVAL_RX} ${cy}`,
    `A ${OVAL_RX} ${OVAL_RY} 0 1 0 ${cx - OVAL_RX} ${cy}`,
    `Z`,
  ].join(' ');

  return (
    // pointerEvents="none" is critical so the overlay never blocks camera touches.
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <Svg height="100%" width="100%">

        {/* Dark overlay with oval punched out — reliable on first render */}
        <Path
          d={overlayPath}
          fill="rgba(0,0,0,0.60)"
          fillRule="evenodd"
        />

        {/* Oval border — colour and dash reflect centering status */}
        <Ellipse
          cx={cx} cy={cy}
          rx={OVAL_RX} ry={OVAL_RY}
          fill="none"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={strokeDash}
        />

        {/* ✓ checkmark when centred */}
        {showCheck && (
          <SvgText
            x={cx} y={cy}
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
            x={cx} y={cy}
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
