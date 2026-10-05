from __future__ import annotations

import math

from markupsafe import Markup

from exmekh_agent.ui.graph.graph_data import SessionGraphData
from exmekh_agent.ui.resources import render_html, ui_text


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
    return render_html(
        "graph/arrow.svg",
        direction_cls=direction_cls,
        pts=pts,
        color=color,
        opacity=opacity,
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
    parts: list[str] = []

    for history in graph_data.histories:
        is_highlighted = highlighted_history_id == history.id or (
            highlighted_history_id is None and history.is_main
        )
        opacity = "0.90" if (history.is_main or is_highlighted) else "0.78"

        if history.is_main:
            # Main history: thin & dashed horizontal line
            parts.append(
                render_html(
                    "graph/main_history.svg",
                    history_id=history.id,
                    history_x1=history.x1,
                    history_y1=history.y1,
                    history_x2=history.x2,
                    history_y2=history.y2,
                    history_line_color=history.line_color,
                    opacity=opacity,
                )
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
                render_html(
                    "graph/outgoing_history.svg",
                    history_id=history.id,
                    ax_out=ax_out,
                    ay_out=ay_out,
                    bx_out=bx_out,
                    by_out=by_out,
                    history_line_color=history.line_color,
                    stroke_w=stroke_w,
                    opacity=opacity,
                )
            )

            # 2) Incoming / return parallel line (-sep * normal): response exits sub-history B -> back into same attachment point A
            bx_in = round(bx - sep * nx, 2)
            by_in = round(by - sep * ny, 2)
            ax_in = round(ax - sep * nx, 2)
            ay_in = round(ay - sep * ny, 2)
            parts.append(
                render_html(
                    "graph/incoming_history.svg",
                    history_id=history.id,
                    bx_in=bx_in,
                    by_in=by_in,
                    ax_in=ax_in,
                    ay_in=ay_in,
                    history_line_color=history.line_color,
                    stroke_w=stroke_w,
                    opacity=opacity,
                )
            )

            # Turnaround cap at outer end B connecting outgoing line to return line
            parts.append(
                render_html(
                    "graph/turnaround.svg",
                    bx_out=bx_out,
                    by_out=by_out,
                    control_x=round(bx + 12.0 * ux, 2),
                    control_y=round(by + 12.0 * uy, 2),
                    bx_in=bx_in,
                    by_in=by_in,
                    history_line_color=history.line_color,
                    stroke_w=stroke_w,
                    opacity=opacity,
                )
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
                render_html(
                    "graph/interaction.svg",
                    history_interaction_x=history.interaction_x,
                    history_interaction_y=history.interaction_y,
                    history_line_color=history.line_color,
                )
            )

        # Endpoint labels for the history straight line
        if history.is_main:
            lbl_x = history.x1 + 6.0
            lbl_y = history.y1 - 10.0
            angle_label = ui_text("graph.graph_svg.main_history")
        else:
            lbl_x = max(
                24.0, min(max(24.0, graph_data.width - 190.0), history.x2 - 45.0)
            )
            lbl_y = max(
                18.0, min(max(18.0, graph_data.height - 15.0), history.y2 - 10.0)
            )
            angle_label = f"{history.label} ({int(history.angle_deg)}°)"

        parts.append(
            render_html(
                "graph/history_label.svg",
                lbl_x=lbl_x,
                lbl_y=lbl_y,
                history_line_color=history.line_color,
                angle_label=angle_label,
            )
        )

        # Render downward counterclockwise skill branches for this history (thin & dashed)
        for branch in history.skill_branches:
            if len(branch.points) >= 2:
                pts_str = " ".join(f"{p[0]},{p[1]}" for p in branch.points)
                parts.append(
                    render_html(
                        "graph/skill_branch.svg",
                        branch_skill_name=branch.skill_name,
                        pts_str=pts_str,
                        branch_color=branch.color,
                    )
                )
                parts.append(
                    render_html(
                        "graph/skill_label.svg",
                        branch_label_x=branch.label_x,
                        label_y=branch.label_y + 14.0,
                        branch_color=branch.color,
                        branch_skill_name=branch.skill_name,
                        branch_depth=branch.depth,
                    )
                )

        # Draw SVG circles at each node position
        for node in history.nodes:
            r = "8" if node.id == selected_node_id else "5.5"
            parts.append(
                render_html(
                    "graph/node.svg",
                    node_x=node.x,
                    node_y=node.y,
                    r=r,
                    node_color=node.color,
                )
            )

    return render_html(
        "graph/canvas.svg",
        width=int(graph_data.width),
        height=int(graph_data.height),
        content=Markup("\n".join(parts)),
    )
