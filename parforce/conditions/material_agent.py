"""Rewrite, construct, and visually review first-frame physical conditions offline."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .action_rewrite import POLICY as REWRITE_POLICY, compile_rigid_action, rewrite_case
from .firstview_v2 import digest
from .run import worker, write_json

ROOT = Path(__file__).resolve().parent
PLANNING = """Return one compact JSON object with the specified material and
fields. Coordinates are pixels normalized over the full image to 0..1000.
Trace what is visible without changing the subject or action.
An alignment overlay, when provided, marks the previous proposal. Inspect its
landmark errors against the separate original first frame before replanning.
If the route is unsupported, return {"material":"unsupported","reason":"short
concrete explanation"}. Return JSON only, with no narration or trajectories.
"""

LIQUID_PLANNING = """Liquid pour: return material liquid_pour, rim [left,top,right,bottom] for the narrow
mouth ellipse of the receiver, base [left,top,right,bottom] for the lower foot,
nozzle_xy [x,y] from the visible outlet or a specified offscreen source,
initial_fill as a visible fraction of receiver height and strength gentle,
moderate or strong. Include already_pouring as a boolean, true only if there
is a visible connected stream spanning the air gap in the first frame. For
a source out of frame above, use a nozzle at y=0 above the receiver center;
if a connected stream enters the top border, use that entry point instead.
Check whether the photographed gap is empty. Liquid inside the pitcher or the
receiver alone is insufficient evidence of active pouring. Preserve a partially
filled receiver before adding new liquid. If a cropped pitcher is visible at one
side, use its visible lip facing the receiver, with an appropriate initial pour
pose, rather than the distant water surface, image corner, or pitcher handle.
A cropped receiver can have a rim just above the visible viewport. Extrapolate
only with visible wall evidence, add this as an assumption, and allow negative
y coordinates down to -180 for the cropped rim and down to -300 for the outlet.
A visible first-frame stream also needs stream_rim_xy, the point where it
crosses the plane of the receiver mouth. Use null before a pour has started.
A visible oblique outlet must stay aligned with the observed source. A crooked
jet may curve under gravity. Give the interior liquid level where visible,
including a partially filled glass. Treat a tinted heavy base as glass unless
a separate liquid boundary is visible. Exclude handles, reflections, the tabletop
and background jars from rim and base boxes. Give tight rim ellipses: keep the
vertical size close to the actual visible lip, not the whole glass.
If pouring has not started and the source must move to reach the opening,
place nozzle_xy at a reachable pouring pose above the receiver and explain
the source movement as a plan assumption. Avoid a nearly horizontal jet
across the entire frame. Preserve a stream already present in the first frame.
Give body_mid_x as [left_x,right_x] at the middle of the receiver height
to capture bulging or tapered walls. Use null when both edges are obscured.
Give handle_box around a visible receiver handle only, otherwise null. Keep it
separate from the main vessel boxes.
"""

CLOTH_GRIP_PLANNING = """Return exactly these four keys: material, contour,
grips, fixed_xy. The material value is cloth_grip. Trace 6 to 12 ordered
outline points [x,y], covering the entire garment, with no duplicate points.
Format moving grips as [x,y,phase_index,action_index] using zero-based indexes
of the frozen actions. Only pull, lift and fold need a grip. Place a grip at
each named corner. Put fixed attachment points [x,y] in fixed_xy, or use []
when there is no attachment. Never add an explanation inside the arrays.
"""

CLOTH_WIND_PLANNING = """Return material cloth_wind and bands, a list of 6 to
12 rows [y,left_x,right_x] tracing both visible sides of the named curtain from
its attached top down to its free edge or the bottom of the crop. Each row is
one horizontal cross-section at a different y. Follow the fabric itself, using
its colour and folds. Exclude the window, walls, and other curtains. Also return
direction toward_camera, away_from_camera, left or right, and strength gentle,
moderate or strong.
Resolve wind direction in the visible room coordinates: wind through a window
behind the curtain is normally toward the camera in an indoor viewpoint.
Fix only the attached upper edge.
"""
ROUTE_PLANNING = {"cloth_grip": CLOTH_GRIP_PLANNING, "cloth_wind": CLOTH_WIND_PLANNING,
                  "liquid_pour": LIQUID_PLANNING, "rigid": ""}
REVIEW = """The image shows the real first frame and five sampled times of GENERATED
PSS and SDS. Return strict JSON with "pass" true only when all properties hold:
(1) the reconstructed initial geometry follows the photographed target, (2) the
motion responds in the instructed direction and at the correct locations, (3)
the condition maintains identity, attachment and contact without obvious
interpenetration or duplicate subjects, and (4) both conditions are legible.
For a sustained active pour, inspect filling across times and an attached
stream of plausible width. For cloth, inspect fixed attachments, contact, local
folds and material continuity. Include "checks" with booleans named alignment,
action, physics and legibility. Cite a relevant visible time in the short
"reason" (at most 25 words), and give the earliest responsible "stage" from alignment, action,
simulation or rendering. If pass, use stage "none". Judge the generated
conditions, not an unseen real future."""


def pixel_tuple(value, size, minimum_y=0):
    if not isinstance(value,list) or len(value)!=size or any(
            type(x) not in (int,float) or not math.isfinite(x)
            or not (minimum_y if index%2 else 0)<=x<=1000 for index,x in enumerate(value)):
        raise ValueError('Expected bounded full-image coordinates')
    scale = [384/1000, 256/1000]*(size//2)
    return [round(x*s) for x,s in zip(value,scale,strict=True)]


def bbox(value, minimum_y=0):
    x0,y0,x1,y1 = pixel_tuple(value,4,minimum_y=minimum_y)
    if x1-x0<6 or y1-y0<2:
        raise ValueError('Invalid rim or base extent')
    return [x0,y0,x1,y1]


def parse(raw):
    clean = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.I).strip()
    result, _ = json.JSONDecoder().raw_decode(clean)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def initial_scene(prompt):
    return re.split(r'\bexternal force\s*:', prompt, maxsplit=1, flags=re.I)[0]


def prefilled_receiver(prompt):
    receiver = r'(?:glass|cup|tumbler|mug)'
    return re.search(receiver + r'\s+(?:part(?:ly|ially) filled|containing\s+(?:\w+\s+){0,3}liquid)',
                     initial_scene(prompt),flags=re.I) is not None


def validate_receiver_fill(prompt, fill):
    empty = re.search(r'\bempty\s+(?:[\w-]+\s+){0,4}(?:glass|cup|tumbler|mug)\b',initial_scene(prompt),flags=re.I)
    if prefilled_receiver(prompt) and fill == 0:
        raise ValueError('The initial caption describes liquid already in the receiver. Place its meniscus and preserve the fill fraction before inflow.')
    if empty and fill > .03:
        raise ValueError('The initial caption describes an empty receiver. Distinguish its glass base from a liquid meniscus.')


def tighten_upright_receiver_rim(first, rim, base, shallow=False):
    """Refine an overtall lip box when the source image has a distinct upper edge band."""
    left, top, right, bottom = rim
    width = right-left
    min_ratio = .12 if shallow else .28
    if top < 5 or bottom-top < min_ratio*width or (not shallow and base[1]-top < .9*width):
        return None
    gray = cv2.cvtColor(np.asarray(first),cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray,(3,3),.65)
    vertical_edges = np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
    profile = np.median(vertical_edges[:,left+width//5:right-width//5],axis=1)
    band = max(4,round(.075*width))
    upper = max(2,top-band)
    peak = upper+int(np.argmax(profile[upper:top+band+1]))
    if profile[peak] < 16:
        return None
    limit = min(bottom,peak+max(7,round(.20*width)))
    strong_rows = np.flatnonzero(profile[peak:limit] >= .54*profile[peak])+peak
    last = peak
    for row in strong_rows:
        if row-last > band:
            break
        last = int(row)
    refined_top = max(upper,peak-2)
    refined_bottom = max(last+2,refined_top+round(.055*width)) if shallow else last+2
    refined = [left,refined_top,right,refined_bottom]
    if refined[3]-refined[1] > bottom-top-max(4,round(.06*width)):
        return None
    return refined


def visible_tapered_foot(first, rim, base, prompt):
    """Use a clear cup's colored contents to bound its narrowing lower cavity."""
    scene = initial_scene(prompt)
    if not re.search(r'\b(?:containing|filled with)\s+(?:amber|red|orange|dark|colou?red)\s+liquid\b',
                     scene,flags=re.I):
        return None
    left,_,right,lip_bottom = rim
    _,top,_,bottom = base
    width = right-left
    if base[2]-base[0] < .82*width or top < lip_bottom+10 or width < 30:
        return None
    hsv = cv2.cvtColor(np.asarray(first),cv2.COLOR_RGB2HSV)
    body = hsv[lip_bottom:bottom,left+3:right-3]
    mask = np.uint8((body[...,1] >= 145) & (body[...,2] >= 85))
    _,labels,stats,_ = cv2.connectedComponentsWithStats(mask)
    core = labels[:,labels.shape[1]//3:2*labels.shape[1]//3]
    occupied = core[core>0]
    if len(occupied) < .06*mask.size:
        return None
    component = np.bincount(occupied).argmax()
    if stats[component,cv2.CC_STAT_HEIGHT] < .22*(bottom-lip_bottom):
        return None
    center = round(top+.4*(bottom-top))-lip_bottom
    margin = max(2,round(.025*width))
    edges = []
    for line in labels[max(0,center-3):center+4]:
        columns = np.flatnonzero(line==component)+left+3
        if len(columns) >= .30*width:
            edges.append((columns[0]-margin,columns[-1]+margin))
    if len(edges)<4:
        return None
    a,b = np.rint(np.median(edges,axis=0)).astype(int).tolist()
    if not left+2 < a < b < right-2 or b-a > base[2]-base[0]-.12*width:
        return None
    return [a,top,b,bottom]


def separate_receiver_handle(row):
    """Keep the external handle out of the hollow receiver cross-section."""
    left,top,right,bottom = row['rim']
    hleft,_,hright,_ = row['handle_box']
    width = right-left
    if hleft <= left+.10*width and left+.12*width < hright < left+.45*width:
        left = hright-2
    elif hright >= right-.10*width and right-.45*width < hleft < right-.12*width:
        right = hleft+2
    else:
        return
    row['qwen_rim'] = row['rim']
    row['rim'] = [left,top,right,bottom]
    foot = row['base']
    if foot[2]-foot[0] >= .95*(right-left):
        row['qwen_base'] = foot
        center = (left+right)/2
        half_width = .36*(right-left)
        top = max(foot[1],foot[3]-round(.23*(right-left)))
        row['base'] = [round(center-half_width),top,round(center+half_width),foot[3]]
    if 'middle_width' in row:
        row['middle_width'] = min(row['middle_width'],right-left)
    row['assumptions'].append('The visible loop handle is outside the receiver; its overlapped side is excluded.')


def plan_held_source_outlet(row):
    """Give a held, distant source a reachable tilt pose before the jet begins."""
    scene = initial_scene(row['prompt'])
    if row['already_pouring'] or not re.search(r'\b(?:pitcher|bottle|pouring vessel|pouring container)\b[^.]*\bheld\b',
                                             scene,flags=re.I):
        return
    left,top,right,bottom = row['rim']
    width = right-left
    center_x,center_y = (left+right)/2,(top+bottom)/2
    x,y = row['nozzle_xy']
    dx,dy = x-center_x,center_y-y
    if dy <= 0 or abs(dx) <= max(width,1.9*dy):
        return
    row.setdefault('qwen_nozzle_xy',row['nozzle_xy'])
    row['nozzle_xy'] = [round(center_x+math.copysign(1.55*dy,dx)),y]
    row['assumptions'].append('Before opening flow, the held source tilts to a reachable pose nearer the receiver.')


def coloured_source_edge(first, mouth, nozzle, prompt):
    initial = initial_scene(prompt).lower()
    mouth_x,mouth_y = (mouth[0]+mouth[2])/2,(mouth[1]+mouth[3])/2
    if re.search(r'\b(pitcher|pouring vessel|bottle|pouring container)\b',initial) and abs(nozzle[0]-mouth_x)>30:
        hsv = cv2.cvtColor(np.asarray(first),cv2.COLOR_RGB2HSV)
        y,x = np.indices(hsv.shape[:2])
        side = x>mouth[2]+6 if nozzle[0]>mouth_x else x<mouth[0]-6
        mask = np.uint8((hsv[:,:,1]>100) & (hsv[:,:,2]>70) & (y<mouth_y) & side)
        _,components,stats,_ = cv2.connectedComponentsWithStats(mask)
        for region in np.argsort(stats[:,cv2.CC_STAT_AREA])[::-1]:
            if region == 0 or stats[region,cv2.CC_STAT_AREA]<180:
                continue
            sy,sx = np.nonzero(components == region)
            if np.min((sx-nozzle[0])**2+(sy-nozzle[1])**2)>75**2:
                continue
            edge = np.argmin((sx-mouth_x)**2+(sy-mouth_y)**2)
            return [int(sx[edge]),int(sy[edge])]
    return None


def liquid_plan_image(first, answer, prompt):
    """Show previous projected landmarks on the source image for replanning."""
    try:
        mouth = bbox(answer['rim'],minimum_y=-180)
        foot = bbox(answer['base'])
        nozzle = pixel_tuple(answer['nozzle_xy'],2,minimum_y=-300)
    except (KeyError,TypeError,ValueError):
        return None
    sheet = first.copy()
    labels = ImageDraw.Draw(sheet)
    labels.rectangle(mouth,outline='#00e6fa',width=2)
    labels.rectangle(foot,outline='#00e6fa',width=2)
    labels.ellipse((nozzle[0]-4,nozzle[1]-4,nozzle[0]+4,nozzle[1]+4),outline='#ee3388',width=2)
    labels.text((5,4),'Previous plan | cyan: rim and foot | pink: nozzle',fill='white',stroke_width=1,stroke_fill='black')
    hint = coloured_source_edge(first,mouth,nozzle,prompt)
    if hint is not None:
        qx,qy = hint
        labels.ellipse((qx-4,qy-4,qx+4,qy+4),outline='#ffeb32',width=2)
        labels.text((5,16),f'Yellow: colour edge hint ({round(1000*qx/384)}, {round(1000*qy/256)})',
                    fill='white',stroke_width=1,stroke_fill='black')
    return sheet


def cloth_outline(answer, material):
    if material == 'cloth_wind' and 'bands' in answer:
        bands = answer['bands']
        if not isinstance(bands,list) or not 6 <= len(bands) <= 20 or any(
                not isinstance(row,list) or len(row) != 3 for row in bands):
            raise ValueError('Expected horizontal curtain slices [y,left,right] at distinct heights')
        left,right = [],[]
        for y,x0,x1 in bands:
            a,b = pixel_tuple([x0,y],2),pixel_tuple([x1,y],2)
            if b[0]-a[0]<4 or (left and a[1]<=left[-1][1]):
                raise ValueError('Curtain slices must have positive width and increase in height order')
            left.append(a)
            right.append(b)
        return left + right[::-1]
    contour = coordinate_records(answer['contour'], 2)
    if not isinstance(contour,list) or not 6<=len(contour)<=80:
        raise ValueError('Expected a resolved cloth silhouette')
    return [pixel_tuple(point,2) for point in contour]


def coordinate_records(values, size):
    """Read nested landmark records or an unambiguous flat coordinate list."""
    if isinstance(values, list) and values and all(type(v) in (int, float) for v in values):
        if len(values) % size:
            raise ValueError("Flat landmark list has an incomplete record")
        return [values[index:index + size] for index in range(0, len(values), size)]
    return values


def plan_alignment_image(first, answer, prompt):
    material = answer.get('material','')
    if material == 'liquid_pour':
        return liquid_plan_image(first,answer,prompt)
    if material not in ('cloth_wind','cloth_grip','cloth'):
        return None
    try:
        outline = cloth_outline(answer,material)
    except (KeyError,TypeError,ValueError):
        return None
    sheet = first.copy()
    labels = ImageDraw.Draw(sheet)
    labels.polygon([tuple(point) for point in outline],outline='#00e6fa',width=2)
    for i in range(1,10):
        x,y = round(384*i/10),round(256*i/10)
        labels.line((x,15,x,24),fill='white')
        labels.line((0,y,6,y),fill='white')
        labels.text((x-7,4),str(100*i),fill='white',stroke_width=1,stroke_fill='black')
        labels.text((8,y-4),str(100*i),fill='white',stroke_width=1,stroke_fill='black')
    return sheet


def compile_cloth_grips(answer, boundary, rewrite):
    """Combine grounded grip locations with directions from the action rewrite."""
    plan = rewrite["plan"]
    phases = plan["phases"]
    directions = {"left": [-.19, 0, 0], "right": [.19, 0, 0],
                  "away": [0, .19, 0], "toward_camera": [0, -.19, 0],
                  "up": [0, 0, .19], "down": [0, 0, -.19]}
    ramp = min(1.05, 2.95 / len(phases))
    interval = ramp + .28

    def target_side(label):
        words = set(re.findall(r'[a-z]+', label.lower()))
        if 'left' in words:
            return 'left'
        if 'right' in words:
            return 'right'
        if words & {'lower', 'bottom', 'hem'}:
            return 'hem'
        return None

    movements = [(p, a) for p, phase in enumerate(phases) for a in phase
                 if a['op'] in ('pull', 'lift', 'fold')]
    sides = {target_side(a['target']): (p, a['directions']) for p, a in movements}
    garment_fold = (sides.keys() == {'left', 'right', 'hem'} and not plan['anchors']
                    and sides['left'][0] < sides['hem'][0] and sides['right'][0] < sides['hem'][0]
                    and set(sides['left'][1]) & {'inward', 'right'}
                    and set(sides['right'][1]) & {'inward', 'left'} and 'up' in sides['hem'][1])

    def snap(pixel, label):
        source = np.asarray(pixel_tuple(pixel, 2))
        candidates = boundary
        region = re.sub(r'[_-]+', ' ', label).lower()
        if re.search(r'\b(edge|hem|cuff|corner|corners)\b', region):
            relative = (boundary - boundary.min(0)) / np.maximum(1, np.ptp(boundary, axis=0))
            side = np.ones(len(boundary), dtype=bool)
            if re.search(r'\bleft\b', region):
                side &= relative[:, 0] <= .35
            if re.search(r'\bright\b', region):
                side &= relative[:, 0] >= .65
            if re.search(r'\b(lower|bottom|hem)\b', region):
                side &= relative[:, 1] >= .70
            if re.search(r'\b(upper|top)\b', region):
                side &= relative[:, 1] <= .30
            candidates = boundary[side]
        if not len(candidates):
            raise ValueError("The requested grip region is absent from the photographed contour")
        delta = np.linalg.norm(candidates - source, axis=1)
        nearest = delta.argmin()
        # A named hem or edge can be recovered even when the coarse landmark
        # was placed on the wrong part of a complete first-frame silhouette.
        tolerance = 34 if len(candidates) == len(boundary) else max(34, .40 * np.ptp(boundary, axis=0).max())
        if delta[nearest] > tolerance:
            raise ValueError("Proposed grip or anchor is too far from the photographed edge")
        return candidates[nearest].tolist()

    moving = {(p, a) for p, phase in enumerate(phases) for a, action in enumerate(phase)
              if action["op"] in ("pull", "lift", "fold")}
    anchor_pixels = coordinate_records(answer.get("fixed_xy", []), 2)
    if not isinstance(anchor_pixels, list) or len(anchor_pixels) > 8 or bool(anchor_pixels) != bool(plan["anchors"]):
        raise ValueError("Ground only fixed regions named by the rewrite")
    fixed = [snap(xy, plan["anchors"][0] if len(plan["anchors"]) == 1 else '') for xy in anchor_pixels]
    grip_rows = coordinate_records(answer["grips"], 4)
    if not isinstance(grip_rows, list) or not 1 <= len(grip_rows) <= 8:
        raise ValueError("Expected one to eight local contact patches")
    grips, covered = [], set()
    return_phase = next((i for i, phase in enumerate(phases) if any(a["op"] == "return" for a in phase)), None)
    release_phase = next((i for i, phase in enumerate(phases) if any(a["op"] == "release" for a in phase)), None)
    for row in grip_rows:
        if isinstance(row, list) and len(row) == 4:
            x, y, phase, action = row
            point = [x, y]
        elif isinstance(row, dict):
            phase, action = row["phase"], row["action"]
            point = row["xy"]
        else:
            raise ValueError("A moving grip requires [x,y,phase_index,action_index]")
        if type(phase) is not int or type(action) is not int or (phase, action) not in moving:
            raise ValueError("Each grip must identify its moving phase and action")
        covered.add((phase, action))
        motion = phases[phase][action]
        xy = snap(point, motion['target'])
        delta = np.zeros(3)
        for direction in motion["directions"]:
            if direction in directions:
                delta += directions[direction]
            elif direction in ("inward", "toward_target"):
                if direction == "toward_target" and not fixed:
                    raise ValueError("Moving toward a target edge requires grounded fixed patches")
                center = np.mean(fixed if fixed else boundary, axis=0)
                ray = center - xy
                ray[1] *= -1
                delta[:2] += .19 * ray / max(float(np.linalg.norm(ray)), 1e-6)
            else:
                raise ValueError("Grip direction is not executable")
        if np.linalg.norm(delta) < 1e-4:
            raise ValueError("No movement for the requested grip")
        if motion["op"] == "fold" and delta[2] <= 0:
            delta[2] = .008
        grip = dict(xy=xy, translation_fraction=delta.tolist(), start=.22 + phase * interval,
                    ramp=ramp, lift_arc=.09 if motion["op"] == "fold" or delta[2] > 0 else .05,
                    return_start=None if return_phase is None else .22 + return_phase * interval,
                    release_at=None if release_phase is None else .22 + release_phase * interval)
        side = target_side(motion['target'])
        single_half_fold = (len(movements) == 1 and return_phase is None and not fixed
                            and ((side == 'left' and set(motion['directions']) == {'up','right'}) or
                                 (side == 'right' and set(motion['directions']) == {'up','left'})))
        if garment_fold:
            axis, hinge = {'left': (0, .36), 'right': (0, .64), 'hem': (1, .47)}[side]
            grip['fold'] = dict(axis=axis, hinge_fraction=hinge, angle_rad=math.pi,
                                clearance_m=.018 if side == 'hem' else .009,
                                hem_grasp=side == 'hem')
        elif single_half_fold or (motion['op'] == 'fold' and np.linalg.norm(delta[:2]) > 1e-4):
            axis = int(np.argmax(np.abs(delta[:2])))
            grip['fold'] = dict(axis=axis, hinge_fraction=.5, angle_rad=math.pi, clearance_m=.010)
        grips.append(grip)
    if covered != moving:
        raise ValueError("The geometry response skipped a rewritten moving action")
    return grips, fixed


def prepare_plan(source, answer, rewrite=None):
    material = answer["material"]
    if material == 'cloth' and 'grips' in answer and 'strength' not in answer:
        material = 'cloth_grip'
    if material == 'cloth' and {'direction', 'strength'} <= answer.keys() and 'grips' not in answer:
        material = 'cloth_wind'
    if material=="unsupported":
        raise ValueError(answer.get("reason","Unsupported action"))
    assumptions = answer.get('assumptions',[])
    if not isinstance(assumptions,list) or not all(isinstance(x,str) for x in assumptions):
        raise ValueError('Assumptions must be a list of strings')
    image = Path(source['image']).resolve()
    row = dict(case=source['case'],image=str(image),prompt=source['prompt'],
               first_frame_sha256=digest(image),material=material,assumptions=assumptions)
    if rewrite is not None:
        if material != rewrite['plan']['route']:
            raise ValueError('Geometry changed the rewritten material route')
        row['action_rewrite'] = rewrite
    strength = {'gentle':.72,'moderate':1.,'strong':1.30}
    if material=='liquid_pour':
        fill = answer['initial_fill']
        if type(fill) not in (int,float) or not 0<=fill<=.92:
            raise ValueError('Unsupported initial fill')
        validate_receiver_fill(source['prompt'],fill)
        active = answer['already_pouring']
        if type(active) is not bool:
            raise ValueError('The first-frame stream indicator must be a boolean')
        row.update(rim=bbox(answer['rim'],minimum_y=-180),base=bbox(answer['base']),
                   nozzle_xy=pixel_tuple(answer['nozzle_xy'],2,minimum_y=-300),initial_fill=fill,
                   already_pouring=active,
                   particles_per_emission=round(13*strength[answer['strength']]))
        if rewrite is not None:
            control = rewrite['plan']
            flow_strength = 'strong' if control['strength'] == 'firm' else control['strength']
            row['particles_per_emission'] = round(13 * strength[flow_strength])
            if control.get('stop_condition') == 'before_rim':
                row['liquid_stop_condition'] = 'before_rim'
        with Image.open(image) as first:
            rgb = first.convert('RGB')
        tapered_foot = visible_tapered_foot(rgb,row['rim'],row['base'],source['prompt'])
        if tapered_foot is not None:
            row['qwen_base'] = row['base']
            row['base'] = tapered_foot
            row['assumptions'].append('The colored liquid inside the lower glass constrains the tapered foot.')
        refined_rim = tighten_upright_receiver_rim(rgb,row['rim'],row['base'],shallow=tapered_foot is not None)
        if refined_rim is not None:
            row['qwen_rim'] = row['rim']
            row['rim'] = refined_rim
            row['assumptions'].append('A coherent upper lip edge band refines the wide rim box.')
        if not active and re.search(r'\b(juice|wine|coffee|tea|cola)\b',source['prompt'],flags=re.I):
            edge = coloured_source_edge(rgb,row['rim'],row['nozzle_xy'],source['prompt'])
            if edge is not None:
                row['qwen_nozzle_xy'] = row['nozzle_xy']
                row['nozzle_xy'] = edge
                row['assumptions'].append('Visible coloured-liquid edge facing the receiver constrains the pouring pose.')
        if active and answer.get('stream_rim_xy') is not None:
            row['stream_rim_xy'] = pixel_tuple(answer['stream_rim_xy'],2)
        middle = answer.get('body_mid_x')
        if middle is not None:
            if not isinstance(middle,list) or len(middle)!=2:
                raise ValueError('Expected left and right mid-height edge coordinates')
            a,b = [pixel_tuple([x,0],2)[0] for x in middle]
            if b-a<6:
                raise ValueError('Mid-height wall width is invalid')
            row['middle_width'] = b-a
        if answer.get('handle_box') is not None:
            row['handle_box'] = bbox(answer['handle_box'])
            separate_receiver_handle(row)
        plan_held_source_outlet(row)
    elif material in ('cloth_wind','cloth_grip'):
        row['contour'] = cloth_outline(answer,material)
        if material=='cloth_wind':
            from .cloth import refine_grounded_contour

            frame = cv2.imread(str(image))
            visible_rows = np.flatnonzero(frame.max((1,2))>3)
            for point in row['contour']:
                point[1] = int(np.clip(point[1],visible_rows[0],visible_rows[-1]))
            if len({y for _,y in row['contour']})<4:
                raise ValueError('Trace the curtain sides at several different heights, from its attachment to its visible lower edge.')
            row['qwen_coarse_contour'] = row['contour']
            row['contour'], _ = refine_grounded_contour(frame,row['contour'],hanging=True)
            axis = {'toward_camera':(0,-1,0),'away_from_camera':(0,1,0),
                    'left':(-1,0,0),'right':(1,0,0)}
            wind_direction = answer['direction']
            if rewrite:
                direction = rewrite['plan']['phases'][0][0]['directions']
                if direction in (['left'], ['right'], ['toward_camera'], ['away']):
                    wind_direction = {'away':'away_from_camera'}.get(direction[0], direction[0])
            row['wind_world'] = (1.4*strength[answer['strength']]*np.asarray(axis[wind_direction])).tolist()
        else:
            from .cloth import refine_grounded_contour

            motion = {'left':[-.19,0,.008],'right':[.19,0,.008],'away':[0,.19,.06],
                      'toward':[0,-.19,.06],'up':[0,.15,.19]}
            row['qwen_coarse_contour'] = row['contour']
            row['contour'], boundary = refine_grounded_contour(cv2.imread(str(image)), row['contour'])
            if rewrite is not None:
                row['grips'], row['fixed_xy'] = compile_cloth_grips(answer, boundary, rewrite)
                return row
            if not 1<=len(answer['grips'])<=4:
                raise ValueError('Expected one to four grip patches')
            row['grips'] = []
            for grip in answer['grips']:
                source_point = np.asarray(pixel_tuple(grip['xy'], 2))
                delta = np.linalg.norm(boundary - source_point, axis=1)
                nearest = delta.argmin()
                if delta[nearest] > 34:
                    raise ValueError('Proposed grip is too far from the photographed garment edge')
                row['grips'].append(dict(xy=boundary[nearest].tolist(),
                    qwen_xy=source_point.tolist(), translation_fraction=motion[grip['direction']],
                    start={'first':.22,'second':1.55}[grip['phase']],ramp=1.05,lift_arc=.09))
    else:
        raise ValueError('Unknown material')
    return row


def model_request(model, processor, images, prompt, budget, progress):
    import torch
    from transformers import StoppingCriteria

    content = [{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt}]
    inputs = processor.apply_chat_template([{'role':'user','content':content}],
        tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True, enable_thinking=False).to(model.get_input_embeddings().weight.device)
    prompt_length = inputs['input_ids'].shape[1]
    started = time.monotonic()
    write_json(progress, {'prompt_tokens': prompt_length, 'generated_tokens': 0})

    class CompleteJson(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            count = input_ids.shape[1] - prompt_length
            if count % 12:
                return False
            write_json(progress, {'prompt_tokens': prompt_length, 'generated_tokens': count,
                                  'seconds': time.monotonic() - started})
            partial = processor.tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True)
            try:
                parse(partial)
            except ValueError:
                return False
            return True

    with torch.inference_mode():
        result = model.generate(**inputs, do_sample=False, max_new_tokens=budget, stopping_criteria=[CompleteJson()])
    write_json(progress, {'prompt_tokens': prompt_length, 'generated_tokens': result.shape[1]-prompt_length,
                          'seconds': time.monotonic()-started, 'finished': True})
    return processor.batch_decode(result[:,prompt_length:],skip_special_tokens=True)[0]


def magnify_receiver(first, rim, base):
    crop = first.crop((max(0,min(rim[0],base[0])-10),max(0,rim[1]-7),
                       min(first.width,max(rim[2],base[2])+10),min(first.height,base[3]+6)))
    return crop.resize((2*crop.width,2*crop.height),Image.Resampling.LANCZOS)


def resolve_receiver_fill(model, processor, first, source, answer, attempt):
    """Separate a real liquid boundary from the solid base using a focused view."""
    prior_fill = answer.get('initial_fill')
    if answer.get('material') != 'liquid_pour' or type(prior_fill) not in (int,float):
        return answer
    scene = initial_scene(source['prompt'])
    prefilled = prefilled_receiver(source['prompt'])
    missing_fill = prefilled and prior_fill == 0
    ambiguous_base = not prefilled and 0 < prior_fill <= .25 and re.search(
        r'\b(base|foot|bottom|tinted glass|colored glass|coloured glass)\b',scene,flags=re.I)
    if not (missing_fill or ambiguous_base):
        return answer
    rim = bbox(answer['rim'],minimum_y=-180)
    foot = bbox(answer['base'])
    crop = magnify_receiver(first,rim,foot)
    crop.save(attempt/'FILL_CROP.jpg')
    if missing_fill:
        minimum = .05
        prompt = ('Estimate the photographed receiver fill before any new stream. The given initial scene '
                  'explicitly states that the cup already contains liquid. The second image magnifies that cup. '
                  'Locate its internal meniscus, distinguish it from the glass foot and reflections, then estimate '
                  'the positive fraction of usable inner height from the floor to the liquid line. Return only '
                  'JSON {"initial_fill": <fraction from 0.05 to 0.92>, "evidence": "short optical cue or assumption"}. '
                  'If the line is faint, state your uncertainty and give a conservative positive estimate.\n'+scene)
    else:
        minimum = 0
        prompt = ('Both images show the same photographed receiver before the requested action. The second '
                  'image magnifies the glass. Look for a separate liquid-air surface spanning the cavity '
                  'above the solid floor. The thick glass foot, tinted base, a reflection in the glass or '
                  'the background seen through it are not liquid. Return 0 when there is no separate liquid '
                  'surface, or the fraction of usable glass height if a meniscus is visible. Return only '
                  'JSON {"initial_fill": <fraction from 0 to 0.92>, "evidence": "visible cue"}.\n'+scene)
    raw = model_request(model,processor,[first,crop],prompt,112,attempt/'FILL_TOKENS.json')
    write_json(attempt/'FILL_RESPONSE.json',{'raw':raw,'prompt':prompt,'prior_fill':prior_fill})
    evidence = parse(raw)
    fill = evidence['initial_fill']
    if type(fill) not in (float,int) or not minimum <= fill <= .92 or not isinstance(evidence['evidence'],str):
        raise ValueError('The focused receiver inspection did not return a bounded fill estimate and evidence')
    return {**answer,'initial_fill':fill,'assumptions':[*answer.get('assumptions',[]),evidence['evidence']]}


def align_cropped_stream(first, limit, mouth, coarse_contact):
    """Trace a long, confirmed jet when its inlet runs out of the photographed frame."""
    left,top,right,_ = limit
    gray = cv2.cvtColor(np.asarray(first),cv2.COLOR_RGB2GRAY)
    visible = np.flatnonzero(gray.max(axis=1)>8)
    if not len(visible) or top > visible[0]+5:
        return None
    upper = int(visible[0])+1
    lower = min(coarse_contact[1],mouth[3],first.height-1)
    span = lower-upper
    if span < 34:
        return None
    smooth = cv2.GaussianBlur(gray[upper:lower+1,left:right],(3,3),.65)
    edges = cv2.Canny(smooth,8,20)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,max(8,round(.20*span)),
                            minLineLength=round(.45*span),maxLineGap=5)
    if lines is None:
        return None
    pairs = []
    radius = max(11,.16*(mouth[2]-mouth[0]))
    for a,y0,b,y1 in lines[:,0]:
        delta = y1-y0
        if abs(delta) < .35*span:
            continue
        slope = (b-a)/delta
        entry = a-slope*y0+left
        contact = entry+slope*span
        if left <= entry < right and abs(contact-coarse_contact[0]) <= radius:
            pairs.append((entry,contact))
    if len(pairs)<2 or np.ptp(np.asarray(pairs),axis=0).max()>radius:
        return None
    inlet_x,rim_x = np.median(pairs,axis=0)
    inlet_band = gray[upper-1:upper+7,max(left,round(inlet_x-radius)):min(right,round(inlet_x+radius)+1)]
    lateral = np.abs(cv2.Sobel(cv2.GaussianBlur(inlet_band,(3,3),.65),cv2.CV_32F,1,0,ksize=3))
    center_row = np.median(lateral[1:4],axis=0)
    if center_row.max() > 20:
        inlet_x = max(left,round(inlet_x-radius))+int(np.argmax(center_row))
    return ((inlet_x,upper),(rim_x,lower))


def align_short_bright_stream(first, mouth, entry, contact):
    """Trace the ridge of a confirmed short, opaque stream against its surroundings."""
    upper,lower = round(entry[1])+5,mouth[1]+4
    if not 12 <= lower-upper <= 30 or upper < 0 or lower >= first.height:
        return None
    radius = round(.25*(mouth[2]-mouth[0]))
    left = max(4,math.floor(min(entry[0],contact[0]))-radius)
    right = min(first.width-4,math.ceil(max(entry[0],contact[0]))+radius+1)
    if right-left < 12:
        return None
    gray = cv2.cvtColor(np.asarray(first),cv2.COLOR_RGB2GRAY).astype(np.float32)[upper:lower+1]
    ridge = gray[:,left:right]-.5*(gray[:,left-4:right-4]+gray[:,left+4:right+4])
    steps = np.arange(-3,4)
    columns = np.arange(right-left)
    neighbors = columns[None,:]+(3+steps)[:,None]
    score = ridge[0]
    history = []
    for line in ridge[1:]:
        previous = np.pad(score,(3,3),constant_values=-np.inf)[neighbors]-2*abs(steps)[:,None]
        winner = previous.argmax(axis=0)
        score = line+previous[winner,columns]
        history.append(steps[winner])
    path = [int(score.argmax())]
    for offsets in reversed(history):
        path.append(path[-1]+int(offsets[path[-1]]))
    path.reverse()
    support = ridge[np.arange(len(path)),path]
    if np.mean(support>=14) < .80 or np.median(support) < 18:
        return None
    return ((left+path[0],upper),(left+path[-1],lower))


def resolve_initial_stream(model, processor, first, answer, attempt, source_prompt=''):
    """Validate a claimed time-zero stream from the visible air gap alone."""
    if answer.get('material') != 'liquid_pour' or answer.get('already_pouring') is not True:
        return answer
    mouth = bbox(answer['rim'],minimum_y=-180)
    x,y = pixel_tuple(answer['nozzle_xy'],2,minimum_y=-300)
    left,top = max(0,min(x,mouth[0])-12),max(0,min(y,mouth[1])-10)
    right,bottom = min(first.width,max(x,mouth[2])+13),min(first.height,max(y,mouth[3])+12)
    gap = first.crop((left,top,right,bottom))
    gap = gap.resize((gap.width*2,gap.height*2),Image.Resampling.LANCZOS)
    gap.save(attempt/'STREAM_CROP.jpg')
    prompt = ('Both images show the same photographed first frame. The second magnifies the proposed '
              'pouring outlet, receiving glass rim and the air gap between them. Look for a visible, '
              'continuous jet that is already spanning the gap in this photo. Liquid in either vessel, '
              'a tilted pitcher, reflections, window edges and liquid merely touching the glass rim do '
              'not show an active jet. A connected stream can enter from a cropped frame boundary. '
              'Require visible stream evidence in the air gap. Return only JSON with '
              'visible_connected_stream (boolean), entry_xy, rim_xy and evidence (at most 15 words). '
              'Use [x,y] normalized to 0..1000 over the SECOND image for both points. entry_xy is '
              'the first visible point of the actual jet, even if the pouring lip lies higher or farther away. '
              'rim_xy is where the jet meets the plane of the receiving cup opening. Use null for '
              'either point that cannot be located.')
    raw = model_request(model,processor,[first,gap],prompt,160,attempt/'STREAM_TOKENS.json')
    write_json(attempt/'STREAM_RESPONSE.json',{'raw':raw,'prompt':prompt,'previous_active':True,
                                               'crop_in_full_frame':[left,top,right,bottom]})
    evidence = parse(raw)
    active = evidence['visible_connected_stream']
    if type(active) is not bool or not isinstance(evidence['evidence'],str):
        raise ValueError('The magnified first-frame stream check needs visual evidence and a boolean')
    revised = {**answer,'already_pouring':active,'assumptions':[*answer.get('assumptions',[]),evidence['evidence']]}
    if not active:
        revised.pop('stream_rim_xy',None)
    else:
        coordinates = []
        for key in ('entry_xy','rim_xy'):
            point = evidence.get(key)
            if (not isinstance(point,list) or len(point) != 2 or
                    not all(type(v) in (int,float) and 0 <= v <= 1000 for v in point)):
                break
            coordinates.append((left+point[0]*(right-left)/1000,top+point[1]*(bottom-top)/1000))
        coarse = answer.get('stream_rim_xy')
        image_fit = None
        fit_evidence = None
        contact_candidates = []
        if len(coordinates) == 2 and mouth[1]-6 <= coordinates[1][1] <= mouth[1]+.18*(mouth[2]-mouth[0]):
            contact_candidates.append(tuple(round(v) for v in coordinates[1]))
        if coarse is not None:
            contact_candidates.append(pixel_tuple(coarse,2))
        for contact_guess in contact_candidates:
            image_fit = align_cropped_stream(first,(left,top,right,bottom),mouth,contact_guess)
            if image_fit is not None:
                fit_evidence = 'Long-edge evidence in the source image locates the clipped jet.'
                break
        if image_fit is None and len(coordinates) == 2 and re.search(r'\bmilk\b',source_prompt,flags=re.I):
            image_fit = align_short_bright_stream(first,mouth,*coordinates)
            if image_fit is not None:
                fit_evidence = 'The first-frame bright ridge locates the short opaque stream.'
        if image_fit is not None:
            coordinates = image_fit
        if len(coordinates) == 2:
            inlet,contact = coordinates
            if (inlet[1]+3 < contact[1] and mouth[0]-5 <= contact[0] <= mouth[2]+5 and
                    mouth[1]-8 <= contact[1] <= mouth[3]+8):
                revised['nozzle_xy'],revised['stream_rim_xy'] = [
                    [round(u*1000/first.width),round(v*1000/first.height)] for u,v in coordinates]
                if fit_evidence is not None:
                    revised['assumptions'].append(fit_evidence)
    return revised


def review_image(case):
    image = Image.new('RGB',(960,282),(20,25,28))
    labels = ImageDraw.Draw(image)
    for i,name in enumerate(('particles','depth')):
        video = cv2.VideoCapture(str(case/f'{name}.mp4'))
        try:
            for j,frame in enumerate((0,9,20,34,48)):
                video.set(cv2.CAP_PROP_POS_FRAMES,frame)
                ok,bgr = video.read()
                if not ok:
                    raise ValueError(f'Could not decode {name} at frame {frame}')
                rgb = cv2.cvtColor(cv2.resize(bgr,(192,125)),cv2.COLOR_BGR2RGB)
                image.paste(Image.fromarray(rgb),(j*192,i*141+16))
                labels.text((j*192+4,i*141+2),f'{name}  {frame/12:.2f}s',fill='white')
        finally:
            video.release()
    return image


def liquid_review_detail(case):
    """Magnify actual particle frames and attach measured receiver occupancy."""
    sampling = (0,9,20,34,48)
    with np.load(case/'states.npz') as states:
        ys,xs = np.nonzero(states['depth_z'][0] > 0)
        finite = np.isfinite(states['positions'][list(sampling)]).all(axis=2)
        inside = (states['captured'][list(sampling)] & finite).sum(axis=1)
    if not len(xs):
        raise ValueError('No visible initial receiver for liquid review')
    x0,x1 = max(0,int(xs.min())-8),min(384,int(xs.max())+9)
    y0,y1 = max(0,int(ys.min())-6),min(256,int(ys.max())+7)
    sheet = Image.new('RGB',(960,188),(20,25,28))
    labels = ImageDraw.Draw(sheet)
    video = cv2.VideoCapture(str(case/'particles.mp4'))
    try:
        for column,frame in enumerate(sampling):
            video.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,bgr = video.read()
            if not ok:
                raise ValueError(f'Could not decode particle detail at frame {frame}')
            image = Image.fromarray(cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB))
            image.thumbnail((184,154))
            sheet.paste(image,(column*192+(192-image.width)//2,30))
            labels.text((column*192+6,4),f'{frame/12:.2f}s | {inside[column]} points inside',fill='white')
    finally:
        video.release()
    measured = ', '.join(f'{frame/12:.2f}s={count}' for frame,count in zip(sampling,inside,strict=True))
    return sheet,measured


def curtain_review_detail(case):
    """Show the fixed vertices on the actually rendered curtain at several times."""
    sample = (0,9,20,34,48)
    with np.load(case/'states.npz') as state:
        indices = state['fixed']
        motion = state['vertices'][:,indices]
        drift = np.linalg.norm(motion-motion[0],axis=-1).max()
        camera = motion[list(sample)]@state['rotation'].T + state['offset']
        projection = camera@state['K'].T
        pixels = projection[...,:2]/projection[...,2:]
        _,xs = np.nonzero(np.any(state['depth_z']>0,axis=0))
    x0,x1 = max(0,int(xs.min())-6),min(384,int(xs.max())+7)
    sheet = Image.new('RGB',(960,190),(20,25,28))
    labels = ImageDraw.Draw(sheet)
    video = cv2.VideoCapture(str(case/'depth.mp4'))
    try:
        for column,frame in enumerate(sample):
            video.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,bgr = video.read()
            if not ok:
                raise ValueError(f'Could not decode curtain detail at frame {frame}')
            for x,y in np.rint(pixels[column]).astype(int)[::2]:
                cv2.circle(bgr,(x,y),1,(0,220,255),-1)
            image = Image.fromarray(cv2.cvtColor(bgr[:,x0:x1],cv2.COLOR_BGR2RGB))
            image.thumbnail((184,158))
            sheet.paste(image,(column*192+(192-image.width)//2,24))
            labels.text((column*192+5,4),f'{frame/12:.2f}s | yellow fixed edge',fill='white')
    finally:
        video.release()
    return sheet,float(drift)


def review_prompt(source, plan, first):
    prompt = REVIEW + '\n' + source['prompt']
    if plan['material'] == 'liquid_pour':
        prompt += ('\nBefore judging the rollout, compare initial liquid level and stream presence with the '
                   'original RGB frame. An active stream at time zero needs a visible jet in the photo. '
                   'The controls encode geometry in grayscale. Judge liquid appearance in the later RGB '
                   'video. For a transparent glass, monitor the internal fill using PSS. SDS records the '
                   'first surface, where the static glass wall can occlude the internal liquid.')
        content_rows = np.flatnonzero(np.asarray(first).max((1,2)) > 3)
        if len(content_rows) and plan['nozzle_xy'][1] <= content_rows[0] + 2:
            prompt += ('\nThe inlet is beyond the visible crop. This control represents its flow entering '
                       'the frame edge, with no source container geometry. Check that the visible stream '
                       'runs from the edge to the mouth during inflow. Check its surface in SDS and the '
                       'flow and accumulating fill in PSS; PSS is made of discrete samples.')
    elif plan['material'] == 'cloth_grip':
        prompt += ('\nThis is a supported garment with local moving grips and freely moving edges. '
                   'Evaluate grip order, cloth continuity and support contact.')
    return prompt


def trace(path, case, stage, **evidence):
    record = {'timestamp':datetime.now().astimezone().isoformat(timespec='seconds'),
              'case':case,'stage':stage,**evidence}
    with path.open('a') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n')


def resume_batch(output, cases, identity, action_trace, retry_cases=(), refine_cases=()):
    previous = json.loads((output/'MODEL_READY.json').read_text())
    for field, value in identity.items():
        if previous.get(field) != value:
            raise ValueError(f'Cannot resume with a different {field}')
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
    archive = output/'resume_history'/stamp
    archive.mkdir(parents=True,exist_ok=False)
    (archive/'MODEL_READY.json').write_text(json.dumps(previous,indent=2)+'\n')
    finished = {}
    next_attempt = {}
    recheck = {}
    for source in cases:
        case = source['case']
        directory = output/case
        archived_attempts = (output/'resume_history').glob(f'*/{case}/round-*')
        max_previous_round = max((int(path.name.removeprefix('round-')) for path in archived_attempts
                                  if re.fullmatch(r'round-\d+',path.name)),default=0)
        if max_previous_round:
            next_attempt[case] = max_previous_round+1
        result_path = directory/'RESULT.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result['first_frame_sha256'] != digest(source['image']) or result['prompt'] != source['prompt']:
                raise ValueError(f'Resumed input differs for {case}')
            if result['accepted_by_qwen'] and case not in retry_cases and case not in refine_cases:
                for name, sha in result['artifact_sha256'].items():
                    if digest(output/name) != sha:
                        raise ValueError(f'Resumed artifact differs for {case}: {name}')
                finished[case] = {'accepted_by_qwen':True,'attempt':result['attempt']}
                continue
        if directory.exists():
            attempts = [int(path.name.removeprefix('round-')) for path in directory.iterdir()
                        if re.fullmatch(r'round-\d+',path.name)]
            next_attempt[case] = max(attempts+[max_previous_round])+1
            preserved = directory.rename(archive/case)
            trace(action_trace,case,'preserve_previous_attempts',directory=str(preserved))
            for number in sorted(attempts,reverse=True):
                attempt = preserved/f'round-{number:02}'
                gate_path = attempt/'cases'/case/'PIPELINE_RESULT.json'
                raw_path = attempt/'PLAN_RESPONSE.json'
                if raw_path.is_file() and (case in refine_cases or (
                        gate_path.is_file() and json.loads(gate_path.read_text())['gate']['passed'])):
                    recheck[case] = raw_path
                    break
                rejected = attempt/'REJECTED.json'
                if (rejected.is_file() and raw_path.is_file() and
                        'liquid already in the receiver' in rejected.read_text()):
                    recheck[case] = raw_path
                    break
    return finished,next_attempt,recheck


def batched_decode_experts(module, hidden_states, indices, weights):
    """Batch the selected experts for one decoded token without expanding prefill weights."""
    from transformers.integrations import batched_mm_experts_forward

    if hidden_states.shape[0] == 1:
        return batched_mm_experts_forward(module, hidden_states, indices, weights)
    return type(module).forward.__wrapped__(module, hidden_states, indices, weights)


def load_qwen(model_dir, device_count):
    import torch
    import torch_npu  # initialize Ascend backend
    from transformers import AutoConfig, AutoProcessor, Qwen3_5MoeForConditionalGeneration
    config = AutoConfig.from_pretrained(model_dir,local_files_only=True)
    if config.model_type != 'qwen3_5_moe':
        raise ValueError(f'Unexpected offline model family {config.model_type}')
    config.text_config._experts_implementation = 'eager'
    processor = AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    processor.image_processor.size = {"longest_edge":196608,"shortest_edge":65536}
    # Keep whole decoder layers together. The balanced 48 GiB allocation
    # leaves the language head offloaded when exactly five cards are leased.
    compact_placement = device_count == 5
    placement = 'sequential' if compact_placement else 'auto'
    per_device_memory = '50GiB' if compact_placement else '48GiB'
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_dir,config=config,
        local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',
        device_map=placement,max_memory={i:per_device_memory for i in range(device_count)},low_cpu_mem_usage=True).eval()
    offloaded = [name for name, device in model.hf_device_map.items() if str(device) in ('cpu','disk')]
    if offloaded:
        raise RuntimeError(f'Model placement exceeds the allocated NPU memory: {offloaded}')
    return model,processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--devices',type=int,default=8)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--rewrites',type=Path,help='Precompute rewrites separately and reuse their verified receipts')
    parser.add_argument('--rigid-perception',type=Path,help='First-frame rigid actor and perception records by case')
    parser.add_argument('--batched-npu-experts',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Keep verified completed cases and rerun unresolved cases')
    parser.add_argument('--retry',action='append',default=[],help='Replan a case that failed independent review')
    parser.add_argument('--refine',action='append',default=[],help='Replay a previous plan using the current geometry and review code')
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())
    if not isinstance(cases,list) or not cases:
        parser.error('Expected a nonempty manifest array')
    rigid = json.loads(args.rigid_perception.read_text()) if args.rigid_perception else {}
    if not isinstance(rigid,dict):
        parser.error('Expected a mapping of case identifiers to rigid perception records')
    for row in cases:
        if not isinstance(row,dict) or set(row)!={'case','image','prompt'} or not re.fullmatch(r'[a-zA-Z0-9_-]+',row['case']):
            parser.error('Each row must have only case, first-frame image, and prompt')
        row['image'] = str((args.manifest.parent/row['image']).resolve())
        with Image.open(row['image']) as first:
            if row['case'] not in rigid and first.size!=(384,256):
                parser.error(f"{row['case']} must use the frozen 384x256 viewport")
    for perception in rigid.values():
        for field in ('mask_dir','depth_file'):
            perception[field] = str((args.rigid_perception.parent/perception[field]).resolve())
        if 'background_depth_file' in perception:
            field = 'background_depth_file'
            perception[field] = str((args.rigid_perception.parent/perception[field]).resolve())
    case_ids = {r['case'] for r in cases}
    if len(case_ids)!=len(cases):
        parser.error('Duplicate case identifier')
    revisit_cases = set(args.retry) | set(args.refine)
    if revisit_cases and (not args.resume or not revisit_cases <= case_ids):
        parser.error('Retry and refine require resuming and case identifiers in the manifest')
    if set(args.retry) & set(args.refine):
        parser.error('Choose either retry or refine for each case')
    if args.devices<5 or not 1<=args.rounds<=6:
        parser.error('Invalid device or round count')
    weights = json.loads((args.model/'COMPLETE.json').read_text())
    if weights.get('repo')!='Qwen/Qwen3.5-122B-A10B' or not weights.get('all_weights_sha256_verified'):
        parser.error('Expected a verified local Qwen model')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    cv2.setNumThreads(1)
    rules = ROOT/'policies/material-agent-policy.txt'
    policy = rules.read_text()
    action_trace = args.output/'ACTION_TRACE.jsonl'
    identity = {'commit':weights['commit'],'rules_sha256':digest(rules),'manifest_sha256':digest(args.manifest),
                'rewrite_policy_sha256':digest(REWRITE_POLICY),
                'rigid_perception_sha256':digest(args.rigid_perception) if args.rigid_perception else None}
    if args.resume:
        finished,next_attempt,recheck = resume_batch(args.output,cases,identity,action_trace,args.retry,args.refine)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        finished,next_attempt,recheck = {},{},{}
    load_start = time.monotonic()
    model, processor = load_qwen(args.model,args.devices)
    if args.batched_npu_experts:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        ALL_EXPERTS_FUNCTIONS.register('parforce_npu_batched', batched_decode_experts)
        model.config.text_config._experts_implementation = 'parforce_npu_batched'
    model_record = {**identity,'seconds':time.monotonic()-load_start,
                    'devices':args.devices,'device_map':{k:str(v) for k,v in model.hf_device_map.items()},
                    'pid':os.getpid(),'offline':True,
                    'batched_npu_experts':args.batched_npu_experts,'source_sha256':digest(__file__)}
    write_json(args.output/'MODEL_READY.json',model_record)
    trace(action_trace,None,'model_ready',**model_record)
    batch = {'total':len(cases),'completed':len(finished),'accepted_by_qwen':len(finished),'cases':finished}
    write_json(args.output/'PROGRESS.json',batch)
    for source in cases:
        if source['case'] in finished:
            continue
        first = Image.open(source['image']).convert('RGB')
        source_hash = digest(source['image'])
        route_options = {'rigid'} if source['case'] in rigid else {'cloth_grip','cloth_wind','liquid_pour'}
        rewrite_folder = (args.rewrites or args.output/'rewrites')/source['case']
        try:
            rewrite = rewrite_case(model,processor,first,source,rewrite_folder,route_options)
        except ValueError as error:
            rejected = args.output/source['case']
            rejected.mkdir(parents=True,exist_ok=True)
            result = {'accepted_by_qwen':False,'attempt':0,'notes':str(error),
                      'first_frame_sha256':source_hash,'prompt':source['prompt']}
            write_json(rejected/'RESULT.json',result)
            trace(action_trace,source['case'],'rewrite_rejected',reason=str(error))
            batch['cases'][source['case']] = {'accepted_by_qwen':False,'attempt':0}
            batch['completed'] += 1
            write_json(args.output/'PROGRESS.json',batch)
            continue
        trace(action_trace,source['case'],'rewrite',evidence=str(rewrite_folder/'REWRITE.json'),
              first_frame_sha256=source_hash)
        feedback, previous, landmarks, receiver_detail, accepted = '',None,None,None,False
        if source['case'] in args.retry:
            prior = recheck.pop(source['case'],None)
            if prior is not None:
                prior_answer = parse(json.loads(prior.read_text())['raw'])
                landmarks = plan_alignment_image(first,prior_answer,source['prompt'])
            feedback = ('The prior proposal failed independent review. Re-ground the initial '
                        'state and the point of action in the original first frame.')
        first_attempt = next_attempt.get(source['case'],1)
        for step in range(first_attempt,first_attempt+args.rounds):
            attempt = args.output/source['case']/f'round-{step:02}'
            attempt.mkdir(parents=True)
            plan_type = rewrite['plan']['route']
            planning = '\n\n'.join([policy, PLANNING, ROUTE_PLANNING[plan_type], source['prompt'],
                                      'Frozen action rewrite: '+json.dumps(rewrite['plan']), feedback])
            images = [first] + [image for image in (landmarks,previous,receiver_detail) if image is not None]
            receiver_detail = None
            started = time.monotonic()
            archived_plan = recheck.get(source['case']) if step == first_attempt else None
            if plan_type == 'rigid':
                raw = json.dumps({'material':'rigid'})
                response = {'raw':raw,'first_frame_sha256':source_hash,
                            'planning_source':'supplied first-frame perception and offline rewrite'}
            elif archived_plan is None:
                raw = model_request(model,processor,images,planning,512,attempt/'PLAN_TOKENS.json')
                response = {'raw':raw,'planning_prompt':planning,'first_frame_sha256':source_hash,
                            'own_output_used':previous is not None or landmarks is not None,
                            'seconds':time.monotonic()-started}
            else:
                response = json.loads(archived_plan.read_text())
                if response['first_frame_sha256'] != source_hash:
                    raise ValueError('Archived model answer was produced from a different frame')
                response['reused_after_resume'] = {'path':str(archived_plan),'sha256':digest(archived_plan)}
                raw = response['raw']
            write_json(attempt/'PLAN_RESPONSE.json',response)
            trace(action_trace,source['case'],'replayed_plan' if archived_plan else 'plan',
                  attempt=step,evidence=str(attempt/'PLAN_RESPONSE.json'),
                  first_frame_sha256=source_hash)
            try:
                answer = parse(raw)
                if plan_type == 'rigid':
                    plan = compile_rigid_action(rigid[source['case']],source,rewrite)
                else:
                    landmarks = plan_alignment_image(first,answer,source['prompt'])
                    if landmarks is not None:
                        landmarks.save(attempt/'PLAN_ALIGNMENT.jpg')
                    answer = resolve_receiver_fill(model,processor,first,source,answer,attempt)
                    answer = resolve_initial_stream(model,processor,first,answer,attempt,source['prompt'])
                    plan = prepare_plan(source,answer,rewrite)
                write_json(attempt/'PLAN.json',plan)
                verdict = worker(plan,attempt)
                trace(action_trace,source['case'],'construct',attempt=step,material=plan['material'],
                      passed=verdict['gate']['passed'],reasons=verdict['gate']['reasons'],
                      evidence=str(attempt/'cases'/source['case']/'PIPELINE_RESULT.json'))
                if not verdict['gate']['passed']:
                    feedback = 'Technical checks failed: '+'; '.join(verdict['gate']['reasons'])
                    if 'Rim or foot fit exceeds 5 pixels' in verdict['gate']['reasons']:
                        receiver_detail = magnify_receiver(first,plan['rim'],plan['base'])
                        receiver_detail.save(attempt/'RECEIVER_FIT_CROP.jpg')
                        feedback += (' The last image magnifies the original receiver. Trace the narrow upper '
                                     'lip and the bottom footprint of the vessel wall itself. Exclude the '
                                     'saucer, handle, thick reflections and any background. A tapered cup '
                                     'has a narrow foot. Return coordinates in the FULL first frame.')
                    try:
                        previous = review_image(attempt/'cases'/source['case'])
                    except ValueError:
                        previous = None
                    if previous is not None:
                        previous.save(attempt/'REVIEW.jpg')
                    continue
                previous = review_image(attempt/'cases'/source['case'])
                previous.save(attempt/'REVIEW.jpg')
                visual_prompt = review_prompt(source,plan,first) + '\nRewritten controls: ' + rewrite['display']
                review_images = [first,previous]
                if plan['material'] == 'liquid_pour':
                    detail, measured = liquid_review_detail(attempt/'cases'/source['case'])
                    detail.save(attempt/'REVIEW_LIQUID_DETAIL.jpg')
                    review_images.append(detail)
                    visual_prompt += ('\nThe final image magnifies the same decoded PSS and labels the actual '
                                      'simulated particle count inside the receiver at each sampled time. '
                                      f'Measured counts: {measured}. Use the close-up to check that the visible '
                                      'particle accumulation is consistent with the counts and direction of flow.')
                elif plan['material'] == 'cloth_wind':
                    detail, drift = curtain_review_detail(attempt/'cases'/source['case'])
                    detail.save(attempt/'REVIEW_CLOTH_ANCHORS.jpg')
                    review_images.append(detail)
                    visual_prompt += ('\nThe condition renders the moving fabric. In the last image, yellow points '
                                      'are actual fixed solver vertices on the photographed upper attachment. '
                                      f'Maximum measured 3D anchor displacement is {drift:.6f} metres. Use the '
                                      'visual overlay to check that folds and free-edge motion stay connected to them.')
                raw_review = model_request(model,processor,review_images,visual_prompt,192,
                                           attempt/'REVIEW_TOKENS.json')
                write_json(attempt/'VISUAL_RESPONSE.json',{'raw':raw_review,'prompt':visual_prompt})
                checked = parse(raw_review)
                if type(checked.get('pass')) is not bool or checked.get('stage') not in (
                        'none','alignment','action','simulation','rendering') or not isinstance(checked.get('reason'),str):
                    raise ValueError('Unexpected visual-review response')
                checks = checked.get('checks')
                if (not isinstance(checks,dict) or set(checks)!={'alignment','action','physics','legibility'}
                        or not all(type(bit) is bool for bit in checks.values())):
                    raise ValueError('Expected all four explicit visual checks')
                accepted = checked['pass'] and all(checks.values()) and checked['stage']=='none'
                write_json(attempt/'VERDICT.json',{**checked,'accepted_by_qwen':accepted,'human_visual_acceptance':None})
                trace(action_trace,source['case'],'visual_review',attempt=step,accepted=accepted,
                      result_stage=checked['stage'],reason=checked['reason'],evidence=str(attempt/'REVIEW.jpg'))
                if accepted:
                    feedback = checked['reason']
                    break
                feedback = f"Previous generated output failed at {checked['stage']}: {checked['reason']}. Revise the plan only where it failed."
            except (ValueError,KeyError,TypeError) as error:
                feedback = 'Invalid or unsupported material plan: '+str(error)
                write_json(attempt/'REJECTED.json',{'error':feedback})
                trace(action_trace,source['case'],'rejected',attempt=step,reason=feedback)
        result = {'accepted_by_qwen':accepted,'codex_visual_acceptance':None,'attempt':step,
                  'notes':feedback,'first_frame_sha256':source_hash,'prompt':source['prompt'],
                  'rewrite_sha256':digest(rewrite_folder/'REWRITE.json')}
        if accepted:
            conditions = attempt/'cases'/source['case']
            first_name = 'FIRST_FRAME.png' if plan_type == 'rigid' else 'first_frame.png'
            evidence = [conditions/name for name in (first_name,'particles.mp4','depth.mp4',
                                                       'states.npz','REPORT.json','PIPELINE_RESULT.json')]
            evidence += [attempt/name for name in ('PLAN.json','VERDICT.json','REVIEW.jpg')]
            evidence += [attempt/name for name in ('PLAN_ALIGNMENT.jpg','FILL_CROP.jpg','FILL_RESPONSE.json',
                                                   'STREAM_CROP.jpg','STREAM_RESPONSE.json',
                                                   'REVIEW_LIQUID_DETAIL.jpg','REVIEW_CLOTH_ANCHORS.jpg')
                         if (attempt/name).is_file()]
            result['artifact_sha256'] = {str(path.relative_to(args.output)):digest(path) for path in evidence}
        write_json(args.output/source['case']/'RESULT.json',result)
        batch['cases'][source['case']] = {'accepted_by_qwen':accepted,'attempt':step}
        batch['completed'] += 1
        batch['accepted_by_qwen'] += int(accepted)
        write_json(args.output/'PROGRESS.json',batch)
        print(datetime.now().isoformat(timespec='seconds'),source['case'],accepted,flush=True)


if __name__=='__main__':
    main()
