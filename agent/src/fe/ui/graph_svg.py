from __future__ import annotations

import html
import math

from fe.ui.graph_data import SessionGraphData


def _build_svg_arrow_polygon(
    *,
    tip_x: float,
    tip_y: float,
    dir_x: float,
    dir_y: float,
    color: str,
    direction_cls: str,
    opacity: str,
) -> str:
    """Build an SVG stealth arrowhead polygon at (tip_x, tip_y) pointing along unit vector (dir_x, dir_y)."""
    arrow_len = 10.0
    half_w = 4.6
    perp_x = -dir_y
    perp_y = dir_x

    left_x = round(tip_x - arrow_len * dir_x + half_w * perp_x, 2)
    left_y = round(tip_y - arrow_len * dir_y + half_w * perp_y, 2)
    notch_x = round(tip_x - 0.65 * arrow_len * dir_x, 2)
    notch_y = round(tip_y - 0.65 * arrow_len * dir_y, 2)
    right_x = round(tip_x - arrow_len * dir_x - half_w * perp_x, 2)
    right_y = round(tip_y - arrow_len * dir_y - half_w * perp_y, 2)

    pts = f"{round(tip_x, 2)},{round(tip_y, 2)} {left_x},{left_y} {notch_x},{notch_y} {right_x},{right_y}"
    return (
        f'<polygon class="graph-sub-arrow {direction_cls}" '
        f'points="{pts}" fill="{color}" stroke="#09090b" stroke-width="0.6" opacity="{opacity}" />'
    )


def render_star_svg_html(
    graph_data: SessionGraphData,
    selected_node_id: str | None = None,
    highlighted_history_id: str | None = None,
) -> str:
    """Render the SVG canvas with:
    - Main history as a thin dashed horizontal line.
    - Sub-tool / sub-agent histories as TWO parallel lines with opposite-direction arrows
      attached at the exact same interaction point on the calling history.
    - Downward counterclockwise skill offset tracks.
    """
    parts: list[str] = [
        f'<svg viewBox="0 0 {int(graph_data.width)} {int(graph_data.height)}" '
        'preserveAspectRatio="none" '
        'class="graph-star-svg w-full h-full absolute inset-0 pointer-events-none">',
        "<defs>",
        '<radialGradient id="constellationGlow" cx="50%" cy="45%" r="55%">',
        '<stop offset="0%" stop-color="#38bdf8" stop-opacity="0.10"/>',
        '<stop offset="55%" stop-color="#a855f7" stop-opacity="0.05"/>',
        '<stop offset="100%" stop-color="#09090b" stop-opacity="0"/>',
        "</radialGradient>",
        "</defs>",
        f'<rect x="0" y="0" width="{int(graph_data.width)}" height="{int(graph_data.height)}" fill="url(#constellationGlow)" />',
    ]

    for history in graph_data.histories:
        is_highlighted = highlighted_history_id == history.id or (
            highlighted_history_id is None and history.is_main
        )
        opacity = "0.90" if (history.is_main or is_highlighted) else "0.78"

        if history.is_main:
            # Main history: thin & dashed horizontal line
            parts.append(
                f'<line class="graph-history-line" data-history-id="{html.escape(history.id)}" '
                f'x1="{history.x1}" y1="{history.y1}" x2="{history.x2}" y2="{history.y2}" '
                f'stroke="{history.line_color}" stroke-width="1.2" '
                f'stroke-dasharray="5,5" opacity="{opacity}" />'
            )
        else:
            # Sub-tool / sub-agent history: TWO parallel lines with arrows in opposite directions
            # flanking the sub-history nodes and attached at the exact same interaction point (interaction_x, interaction_y)
            ax, ay = history.interaction_x, history.interaction_y
            bx, by = history.x2, history.y2
            dx = bx - ax
            dy = by - ay
            seg_len = max(1.0, math.hypot(dx, dy))
            ux = dx / seg_len
            uy = dy / seg_len
            # Unit perpendicular normal
            nx = -uy
            ny = ux
            sep = 11.0
            stroke_w = "1.35" if is_highlighted else "1.15"

            # 1) Outgoing parallel line (+sep * normal): message enters sub-history from attachment point A -> B
            ax_out = round(ax + sep * nx, 2)
            ay_out = round(ay + sep * ny, 2)
            bx_out = round(bx + sep * nx, 2)
            by_out = round(by + sep * ny, 2)
            parts.append(
                f'<line class="graph-history-line graph-sub-parallel-line graph-sub-parallel-out" '
                f'data-history-id="{html.escape(history.id)}" data-direction="out" '
                f'x1="{ax_out}" y1="{ay_out}" x2="{bx_out}" y2="{by_out}" '
                f'stroke="{history.line_color}" stroke-width="{stroke_w}" opacity="{opacity}" />'
            )

            # 2) Incoming / return parallel line (-sep * normal): response exits sub-history B -> back into same attachment point A
            bx_in = round(bx - sep * nx, 2)
            by_in = round(by - sep * ny, 2)
            ax_in = round(ax - sep * nx, 2)
            ay_in = round(ay - sep * ny, 2)
            parts.append(
                f'<line class="graph-history-line graph-sub-parallel-line graph-sub-parallel-in" '
                f'data-history-id="{html.escape(history.id)}" data-direction="in" '
                f'x1="{bx_in}" y1="{by_in}" x2="{ax_in}" y2="{ay_in}" '
                f'stroke="{history.line_color}" stroke-width="{stroke_w}" opacity="{opacity}" />'
            )

            # Turnaround cap at outer end B connecting outgoing line to return line
            parts.append(
                f'<path class="graph-sub-turnaround" '
                f'd="M {bx_out},{by_out} Q {round(bx + 12.0 * ux, 2)},{round(by + 12.0 * uy, 2)} {bx_in},{by_in}" '
                f'fill="none" stroke="{history.line_color}" stroke-width="{stroke_w}" opacity="{opacity}" />'
            )

            # Outgoing arrows along Parallel Line 1 (pointing A -> B, away from attachment point)
            for frac_out in (0.26, 0.56, 0.86):
                t_out = seg_len * frac_out
                parts.append(
                    _build_svg_arrow_polygon(
                        tip_x=ax_out + t_out * ux,
                        tip_y=ay_out + t_out * uy,
                        dir_x=ux,
                        dir_y=uy,
                        color=history.line_color,
                        direction_cls="graph-sub-arrow-out",
                        opacity="0.98",
                    )
                )

            # Incoming / Return arrows along Parallel Line 2 (pointing B -> A, back into the same attachment point)
            for frac_in in (0.76, 0.46, 0.16):
                t_in = seg_len * frac_in
                parts.append(
                    _build_svg_arrow_polygon(
                        tip_x=ax_in + t_in * ux,
                        tip_y=ay_in + t_in * uy,
                        dir_x=-ux,
                        dir_y=-uy,
                        color=history.line_color,
                        direction_cls="graph-sub-arrow-in",
                        opacity="0.98",
                    )
                )

            # Interaction attachment ring on the parent history where both parallel lines meet
            parts.append(
                f'<circle class="graph-interaction-point" '
                f'cx="{history.interaction_x}" cy="{history.interaction_y}" r="14" '
                f'fill="none" stroke="{history.line_color}" stroke-width="1.1" '
                'stroke-dasharray="2,3" opacity="0.82" />'
            )

        # Endpoint labels for the history straight line
        if history.is_main:
            lbl_x = history.x1 + 6.0
            lbl_y = history.y1 - 10.0
            angle_label = "MAIN HISTORY • ORIZZONTALE"
        else:
            lbl_x = max(
                24.0, min(max(24.0, graph_data.width - 190.0), history.x2 - 45.0)
            )
            lbl_y = max(
                18.0, min(max(18.0, graph_data.height - 15.0), history.y2 - 10.0)
            )
            angle_label = f"{history.label} ({int(history.angle_deg)}°)"

        parts.append(
            f'<text x="{lbl_x}" y="{lbl_y}" fill="{history.line_color}" '
            'font-size="10" font-family="monospace" font-weight="600" opacity="0.88">'
            f"{html.escape(angle_label)}</text>"
        )

        # Render downward counterclockwise skill branches for this history (thin & dashed)
        for branch in history.skill_branches:
            if len(branch.points) >= 2:
                pts_str = " ".join(f"{p[0]},{p[1]}" for p in branch.points)
                parts.append(
                    f'<polyline class="graph-skill-branch" data-skill-name="{html.escape(branch.skill_name)}" '
                    f'points="{pts_str}" fill="none" stroke="{branch.color}" '
                    'stroke-width="1.15" stroke-dasharray="3,4" opacity="0.85" />'
                )
                parts.append(
                    f'<text x="{branch.label_x}" y="{branch.label_y + 14.0}" fill="{branch.color}" '
                    'font-size="9.5" font-family="monospace" font-weight="600">'
                    f"↺ skill: {html.escape(branch.skill_name)} (step -{branch.depth})</text>"
                )

        # Draw SVG circles at each node position
        for node in history.nodes:
            r = "8" if node.id == selected_node_id else "5.5"
            parts.append(
                f'<circle cx="{node.x}" cy="{node.y}" r="{r}" fill="{node.color}" '
                'stroke="#09090b" stroke-width="1.5" opacity="0.85" />'
            )

    parts.append("</svg>")
    return "\n".join(parts)
