"""
MindFold 3D — Adaptive Spatial Cognition Assessment and Training Server.

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending. See docs/PATENT_SPECIFICATION.md for claims.

This module implements the Task Presentation Engine, Tiered Distractor
Generation, and Performance Tracking components of the MindFold 3D
invention (Claims 1, 2, 7, 14, 16, 17).

Key inventive endpoints:
  - /get-multiple-choice:  Adaptive trial generation with cognitive profiling (Claim 1a-g)
  - /submit-response:      Per-feature performance tracking (Claim 1h-i)
  - /coaching/*:           LLM-mediated cognitive coaching (Claim 5)
"""
import copy
import math
import random
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Body, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth
import auth_routes
import models
from database import engine, SessionLocal
from shape_features import ShapeFeatureSet
from shape_generation import generate_shape_from_features, generate_mirror_reflection, generate_part_permuted_distractor, analyze_shape_features, canonical_voxel_form, nudge_voxels_until_unique, keep_largest_component
from skeleton_generation import generate_shape_skeleton
from cognitive_mapping import get_difficulty_spec, get_skeleton_spec, perturb_skeleton_spec, SHAPE_DIMENSIONS, TASK_DIMENSIONS, reverse_map_cognitive_profile, cognitive_profile_to_dict
import json
from jose import JWTError, jwt

models.Base.metadata.create_all(bind=engine)

# Canonical viewpoints for perspective-taking mode (Skill 8)
CANONICAL_VIEWPOINTS = [
    {"name": "front",  "position": [0, 0, 8],  "label": "Front"},
    {"name": "back",   "position": [0, 0, -8], "label": "Back"},
    {"name": "right",  "position": [8, 0, 0],  "label": "Right"},
    {"name": "left",   "position": [-8, 0, 0], "label": "Left"},
    {"name": "top",    "position": [0, 8, 0],  "label": "Top"},
    {"name": "bottom", "position": [0, -8, 0], "label": "Bottom"},
]

# Get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)

# Session data for current session
session_data = {
    "total_questions": 0,
    "correct_answers": 0,
    "attempts": [],
    "feature_errors": {}
}

# Persistent data across all sessions
persistent_data = {
    "total_questions": 0,
    "correct_answers": 0,
    "attempts": [],
    "feature_errors": {},
    "feature_stats": {}
}

class ShapeResponse(BaseModel):
    id: str
    voxels: List[List[int]]
    grid_size: List[int]
    features: Optional[dict] = None

class ShapeChoiceSet(BaseModel):
    target: ShapeResponse
    choices: List[ShapeResponse]
    perspective: Optional[dict] = None
    cognitive_profile: Optional[dict] = None

class ResponseSubmission(BaseModel):
    shape_id: str
    correct: bool
    error_type: Optional[str]
    response_time: float
    target_id: str
    target_features: Optional[dict] = None  # Make sure this accepts a dictionary
    task_variant: Optional[str] = None  # "simultaneous", "delayed-short", "delayed-long"
    mirror_mode: Optional[bool] = None  # Whether mirror distractors were included
    interaction_data: Optional[dict] = None  # Layer 3: orientation & orbit interaction metrics

    class Config:
        # Allow arbitrary types for target_features to handle different formats
        arbitrary_types_allowed = True
        # Also allow extra fields for future compatibility
        extra = "allow"
        # Ensure JSON is properly encoded/decoded
        json_encoders = {
            dict: lambda v: v
        }

class CoachingFeedbackRequest(BaseModel):
    """Request for quick feedback after a response."""
    was_correct: bool
    response_time: float
    target_features: Optional[dict] = None
    task_variant: Optional[str] = None
    mirror_mode: Optional[bool] = None
    interaction_data: Optional[dict] = None

class CoachingAdviceRequest(BaseModel):
    """Request for on-demand coaching."""
    user_question: Optional[str] = None

class CoachingResponse(BaseModel):
    """Unified response from all coaching endpoints."""
    coaching_text: str
    source: str  # "llm" or "fallback"
    model: Optional[str] = None
    latency_ms: Optional[float] = None

@app.get("/")
def serve_root():
    return FileResponse("static/index.html")

@app.get("/login")
def serve_login():
    return FileResponse("static/auth.html")

@app.get("/license", response_class=Response)
def serve_license():
    try:
        with open("LICENSE", "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/plain; charset=utf-8")
    except FileNotFoundError:
        return Response(content="License file not found.", media_type="text/plain", status_code=404)

@app.get("/generate-shape")
def generate_example_shape():
    features = ShapeFeatureSet(
        voxel_count=random.randint(5, 10),
        grid_size=(7, 7, 7),
        branching_factor=random.randint(0, 3),
        compactness_score=random.uniform(0.2, 0.8),
        number_of_components=1,
        planarity_score=random.uniform(0.2, 0.9),
        anisotropy_index=random.uniform(0.0, 0.8),
        shape_form_index=random.uniform(-0.8, 0.8),
    )
    shape = generate_shape_from_features(features)
    shape["id"] = f"sample_{uuid.uuid4().hex[:6]}"
    shape["features"] = features.to_dict()
    return shape

@app.post("/generate-shape")
def generate_custom_shape(features: ShapeFeatureSet = Body(...)):
    shape = generate_shape_from_features(features)
    shape["id"] = f"custom_{uuid.uuid4().hex[:6]}"
    return shape



@app.get("/get-multiple-choice", response_model=ShapeChoiceSet)
def get_multiple_choice(
    include_mirror: bool = False,
    include_part_permuted: bool = False,
    perspective_mode: bool = False,
    expert_mode: bool = False,
    current_user: models.User = Depends(auth.get_current_active_user)
):
    print(f"\n=== New Multiple Choice Request (Advanced Generator) ===")
    print(f"User: {current_user.username} (ID: {current_user.id})")
    
    try:
        # 1. Randomize cognitive difficulties using the layered framework
        if expert_mode:
            # Expert mode: shape geometry uses high/expert levels, larger voxel range
            difficulty_levels = ["high", "expert"]
            target_question_voxel_count = random.randint(15, 25)
        else:
            difficulty_levels = ["low", "medium", "high"]
            target_question_voxel_count = random.randint(8, 12)

        # Layer 1: shape geometry dimensions
        shape_difficulties = {
            dim: random.choice(difficulty_levels) for dim in SHAPE_DIMENSIONS
        }

        # Layer 2: task design dimensions (expert mode only affects shape geometry,
        # task features remain user-controlled via separate toggles)
        task_difficulty_levels = ["low", "medium", "high"]
        task_difficulties = {
            dim: random.choice(task_difficulty_levels) for dim in TASK_DIMENSIONS
        }
        # Override task dimensions based on query params
        task_difficulties["mirror_discrimination"] = "high" if include_mirror else "low"
        task_difficulties["configural_binding"] = "high" if include_part_permuted else "low"
        task_difficulties["perspective_taking"] = "high" if perspective_mode else "low"

        # 2. Get combined shape targets + task parameters
        spec = get_difficulty_spec(
            shape_difficulties, task_difficulties,
            target_voxel_count=target_question_voxel_count
        )
        use_mirror = spec.task_params.include_mirror
        use_part_permuted = spec.task_params.include_part_permuted
        print(f"Shape difficulties: {shape_difficulties}")
        print(f"Task params: mirror={use_mirror}, part_permuted={use_part_permuted}, perspective={perspective_mode}, wm={spec.task_params.wm_mode}")

        base_target_sfs = spec.shape_features
        # Ensure target shape is always a single component
        base_target_sfs.number_of_components = 1
        print(f"Target SFS: {base_target_sfs.model_dump(exclude_none=True)}")

        # 3. Generate the target shape using the skeleton-first method
        skeleton_spec = get_skeleton_spec(shape_difficulties, task_difficulties, target_question_voxel_count)
        target_shape_data = generate_shape_skeleton(skeleton_spec)
        print(f"Skeleton spec: archetype={skeleton_spec.archetype}, branches={skeleton_spec.num_branches}, "
              f"loops={skeleton_spec.num_loops}, spread={skeleton_spec.direction_spread}, "
              f"packing={skeleton_spec.packing}, planarity={skeleton_spec.planarity}")

        # Serving-layer orphan cleanup. When the spec requested a single
        # connected component but the skeleton fallback leaked a multi-
        # component result (main shape + stray voxel), keep only the largest
        # connected piece. Specs that legitimately request multi-component
        # shapes are passed through untouched.
        if getattr(skeleton_spec, "num_components", 1) == 1:
            _grid_t = tuple(target_shape_data["grid_size"])
            _largest = keep_largest_component(target_shape_data["voxels"], _grid_t)
            if len(_largest) != len(target_shape_data["voxels"]):
                print(f"Target had multiple components; kept largest "
                      f"({len(_largest)}/{len(target_shape_data['voxels'])} voxels)")
                target_shape_data["voxels"] = [list(v) for v in _largest]
                target_shape_data["features"] = analyze_shape_features(
                    _largest, _grid_t
                ).model_dump(exclude_none=True)

        # The 'features' field includes original targets + calculated actuals
        target_features_dict = target_shape_data["features"]

        # Create the target ShapeResponse instance
        target_shape_response = ShapeResponse(
            id=f"target_{uuid.uuid4().hex[:6]}",
            voxels=target_shape_data["voxels"],
            grid_size=list(target_shape_data["grid_size"]),
            features=target_features_dict
        )
        print(f"Target shape generated. Actual voxels: {target_features_dict.get('voxel_count', 0)}")
        print(f"  Actual compactness: {target_features_dict.get('compactness_score', 0.0):.2f}")
        print(f"  Actual branching: {target_features_dict.get('branching_factor', 0)}")
        print(f"  Actual components: {target_features_dict.get('number_of_components', 0)}")
        print(f"  Actual planarity: {target_features_dict.get('planarity_score', 0.0):.2f}")
        
        # DEBUG: Check if voxels are properly generated
        print(f"DEBUG: Target voxels count: {len(target_shape_data['voxels'])}")
        if len(target_shape_data['voxels']) == 0:
            print("WARNING: Empty voxels array for target!")
            # Generate a simple default shape as fallback
            target_shape_data['voxels'] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            target_shape_response.voxels = target_shape_data['voxels']
        
        # Reverse-map: verify what the generated shape actually embodies
        valid_keys = ShapeFeatureSet.model_fields.keys()
        target_sfs = ShapeFeatureSet(**{k: v for k, v in target_features_dict.items() if k in valid_keys and v is not None})
        verified_profile = reverse_map_cognitive_profile(target_sfs, shape_difficulties)
        cognitive_profile_data = {
            "intended": {
                "shape_difficulties": shape_difficulties,
                "task_difficulties": task_difficulties,
                "recommended_archetype": spec.recommended_archetype,
            },
            "verified": cognitive_profile_to_dict(verified_profile),
        }
        print(f"  Cognitive fidelity: {verified_profile.overall_fidelity:.2f}")

        # Store canonical representation of target shape's voxels for uniqueness checks
        canonical_target_voxels = canonical_voxel_form(target_shape_response.voxels)
        accepted_canonical_voxels_for_question = {canonical_target_voxels}

        distractors = []
        num_distractors = 3  # 3-tier system: Tier 0 (radical), Tier 1 (moderate), Tier 2 (subtle)
        MAX_DISTRACTOR_GENERATION_ATTEMPTS = 10

        # If mirror mode requested, try to generate a mirror distractor first
        if use_mirror:
            mirror_voxels = generate_mirror_reflection(
                target_shape_data["voxels"],
                tuple(target_shape_data["grid_size"])
            )
            if mirror_voxels is not None:
                mirror_canonical = canonical_voxel_form(mirror_voxels)
                if mirror_canonical not in accepted_canonical_voxels_for_question:
                    accepted_canonical_voxels_for_question.add(mirror_canonical)
                    mirror_features = copy.deepcopy(target_features_dict)
                    mirror_item = ShapeResponse(
                        id=f"mirror_{uuid.uuid4().hex[:6]}",
                        voxels=mirror_voxels,
                        grid_size=list(target_shape_data["grid_size"]),
                        features=mirror_features
                    )
                    distractors.append(mirror_item)
                    print(f"Mirror distractor generated successfully")
                else:
                    print(f"Mirror is identical to target (achiral shape), skipping")
            else:
                print(f"Shape is achiral on all axes, no mirror distractor possible")

        # If part-permuted mode requested, try to generate a configural binding distractor
        if use_part_permuted and len(distractors) < num_distractors:
            pp_voxels = generate_part_permuted_distractor(
                target_shape_data["voxels"],
                tuple(target_shape_data["grid_size"])
            )
            if pp_voxels is not None:
                pp_canonical = canonical_voxel_form(pp_voxels)
                if pp_canonical not in accepted_canonical_voxels_for_question:
                    accepted_canonical_voxels_for_question.add(pp_canonical)
                    # Recompute features for the permuted shape (arrangement changes geometry)
                    pp_voxels_set = set(tuple(v) for v in pp_voxels)
                    pp_sfs = analyze_shape_features(pp_voxels_set, tuple(target_shape_data["grid_size"]))
                    pp_features_dict = pp_sfs.to_dict()
                    pp_features_dict["distractor_similarity"] = "part_permuted"
                    pp_item = ShapeResponse(
                        id=f"partperm_{uuid.uuid4().hex[:6]}",
                        voxels=pp_voxels,
                        grid_size=list(target_shape_data["grid_size"]),
                        features=pp_features_dict
                    )
                    distractors.append(pp_item)
                    print(f"Part-permuted distractor generated successfully")
                else:
                    print(f"Part-permuted distractor identical to existing shape, skipping")
            else:
                print(f"Shape not decomposable, no part-permuted distractor possible")

        # Generate remaining distractors via tiered perturbation
        remaining_distractors = num_distractors - len(distractors)
        for i in range(remaining_distractors):
            tier = min(i + len(distractors), 2)  # 0=radical, 1=moderate, 2=subtle
            tier_names = {0: "RADICAL", 1: "MODERATE", 2: "SUBTLE"}
            print(f"--- Generating Distractor {len(distractors)+1} (Tier {tier}: {tier_names.get(tier, '?')}) ---")
            generated_distractor_shape_data = None
            unique_distractor = False

            for attempt in range(MAX_DISTRACTOR_GENERATION_ATTEMPTS):
                print(f"  Attempt {attempt+1}/{MAX_DISTRACTOR_GENERATION_ATTEMPTS}, Tier {tier}")

                distractor_spec = perturb_skeleton_spec(skeleton_spec, tier)
                # Escalate after repeated failures: swap to a different archetype so
                # highly-constrained targets (e.g., a bridge "figure-8") don't keep
                # producing rotational copies.
                if attempt >= 3:
                    alt_archetypes = [a for a in ("tree", "chiral", "bridge")
                                      if a != skeleton_spec.archetype]
                    distractor_spec.archetype = random.choice(alt_archetypes)
                generated_distractor_shape_data = generate_shape_skeleton(distractor_spec)

                if len(generated_distractor_shape_data['voxels']) == 0:
                    print(f"  WARNING: Empty voxels, using fallback")
                    generated_distractor_shape_data['voxels'] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

                # Orphan cleanup on the distractor, only when the distractor
                # spec asked for a single component. Legitimate multi-component
                # distractors are left intact.
                if getattr(distractor_spec, "num_components", 1) == 1:
                    _dgrid = tuple(generated_distractor_shape_data["grid_size"])
                    _dlargest = keep_largest_component(
                        generated_distractor_shape_data["voxels"], _dgrid
                    )
                    if len(_dlargest) != len(generated_distractor_shape_data["voxels"]):
                        print(f"  Distractor had multiple components; kept largest "
                              f"({len(_dlargest)}/{len(generated_distractor_shape_data['voxels'])} voxels)")
                        generated_distractor_shape_data["voxels"] = [list(v) for v in _dlargest]
                        generated_distractor_shape_data["features"] = analyze_shape_features(
                            _dlargest, _dgrid
                        ).model_dump(exclude_none=True)

                distractor_voxels_canonical = canonical_voxel_form(generated_distractor_shape_data["voxels"])
                if distractor_voxels_canonical not in accepted_canonical_voxels_for_question:
                    accepted_canonical_voxels_for_question.add(distractor_voxels_canonical)
                    print(f"  Unique distractor found on attempt {attempt+1}")
                    unique_distractor = True
                    break
                else:
                    print(f"  Identical to existing shape, retrying...")

            if not unique_distractor and generated_distractor_shape_data is not None:
                # Last-ditch: add/remove voxels on the last generated shape
                # until its canonical form is unique.
                print(f"  Nudging voxels to force uniqueness...")
                nudged = nudge_voxels_until_unique(
                    generated_distractor_shape_data["voxels"],
                    accepted_canonical_voxels_for_question,
                    tuple(generated_distractor_shape_data["grid_size"]),
                )
                if nudged is not None:
                    generated_distractor_shape_data["voxels"] = nudged
                    grid_tuple = tuple(generated_distractor_shape_data["grid_size"])
                    nudged_set = {tuple(v) for v in nudged}
                    new_sfs = analyze_shape_features(nudged_set, grid_tuple)
                    generated_distractor_shape_data["features"] = new_sfs.model_dump(exclude_none=True)
                    accepted_canonical_voxels_for_question.add(canonical_voxel_form(nudged))
                    unique_distractor = True
                    print(f"  Nudge succeeded: voxel count now {len(nudged)}")

            if not unique_distractor:
                print(f"  WARNING: Could not generate unique distractor even after nudging; skipping this slot")
                continue

            if generated_distractor_shape_data is None:
                print(f"ERROR: No shape data generated. Skipping.")
                continue

            distractor_features_dict = generated_distractor_shape_data["features"]
            distractor_item = ShapeResponse(
                id=f"distractor_{uuid.uuid4().hex[:6]}",
                voxels=generated_distractor_shape_data["voxels"],
                grid_size=list(generated_distractor_shape_data["grid_size"]),
                features=distractor_features_dict
            )
            distractors.append(distractor_item)
            print(f"Distractor generated. Voxels: {distractor_features_dict.get('voxel_count', 0)}, "
                  f"Compactness: {distractor_features_dict.get('compactness_score', 0.0):.2f}")

        all_choices_responses = distractors + [target_shape_response]
        random.shuffle(all_choices_responses)

        # DEBUG: Check the final output data
        print("\n--- DEBUG: Just before returning from /get-multiple-choice ---")
        print(f"Number of choices: {len(all_choices_responses)}")
        
        # Ensure we're returning exactly 4 choices total (3 distractors + 1 target)
        if len(all_choices_responses) > 4:
            print(f"WARNING: Too many choices generated ({len(all_choices_responses)}), limiting to 4")
            all_choices_responses = all_choices_responses[:4]
        elif len(all_choices_responses) < 4:
            print(f"WARNING: Too few choices generated ({len(all_choices_responses)}), should have 4")
            # If we have too few, we could add simple defaults here, but it's unlikely
        
        # Perspective mode: select viewpoints for target and choices
        perspective_data = None
        if perspective_mode:
            vp_indices = random.sample(range(len(CANONICAL_VIEWPOINTS)), 2)
            target_vp = CANONICAL_VIEWPOINTS[vp_indices[0]]
            choices_vp = CANONICAL_VIEWPOINTS[vp_indices[1]]
            t, c = target_vp["position"], choices_vp["position"]
            dot = sum(a * b for a, b in zip(t, c))
            mag = math.sqrt(sum(a * a for a in t)) * math.sqrt(sum(a * a for a in c))
            angular_distance = round(math.degrees(math.acos(max(-1, min(1, dot / mag if mag > 0 else 0)))))
            perspective_data = {
                "target_viewpoint": target_vp["name"],
                "target_camera_position": target_vp["position"],
                "target_viewpoint_label": target_vp["label"],
                "choices_viewpoint": choices_vp["name"],
                "choices_camera_position": choices_vp["position"],
                "choices_viewpoint_label": choices_vp["label"],
                "angular_distance": angular_distance,
            }
            print(f"Perspective mode: target={target_vp['name']}, choices={choices_vp['name']}, angle={angular_distance}deg")

        response_data = ShapeChoiceSet(target=target_shape_response, choices=all_choices_responses, perspective=perspective_data, cognitive_profile=cognitive_profile_data)
        print(f"Response target voxels count: {len(response_data.target.voxels)}")
        print(f"Response choices count: {len(response_data.choices)}")
        for i, choice in enumerate(response_data.choices):
            print(f"Choice {i+1} voxels count: {len(choice.voxels)}")
        
        return response_data
        
    except Exception as e:
        print(f"ERROR in get_multiple_choice: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a simple fallback response
        target = ShapeResponse(
            id=f"fallback_target_{uuid.uuid4().hex[:6]}",
            voxels=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            grid_size=[3, 3, 3],
            features={}
        )

        choices = [
            ShapeResponse(
                id=f"fallback_distractor_1_{uuid.uuid4().hex[:6]}",
                voxels=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                grid_size=[3, 3, 3],
                features={}
            ),
            ShapeResponse(
                id=f"fallback_distractor_2_{uuid.uuid4().hex[:6]}",
                voxels=[[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                grid_size=[3, 3, 3],
                features={}
            ),
            ShapeResponse(
                id=f"fallback_distractor_3_{uuid.uuid4().hex[:6]}",
                voxels=[[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                grid_size=[3, 3, 3],
                features={}
            ),
            target
        ]
        random.shuffle(choices)

        if len(choices) > 4:
            choices = choices[:4]
            
        print(f"Returning fallback response with target and {len(choices)} choices")
        
        return ShapeChoiceSet(target=target, choices=choices)

@app.post("/submit-response")
def submit_response(data: ResponseSubmission, current_user: models.User = Depends(auth.get_current_active_user)):
    # Debug incoming data
    print(f"\n=== New Response Submission ===")
    print(f"User: {current_user.username} (ID: {current_user.id})")
    print(f"Shape ID: {data.shape_id}, Target ID: {data.target_id}")
    print(f"Correct: {data.correct}, Response Time: {data.response_time}")
    print(f"Target features included: {data.target_features is not None}")
    
    if data.target_features:
        features_count = len(data.target_features) if isinstance(data.target_features, dict) else "Not a dict"
        print(f"Features received count: {features_count}")
        if isinstance(data.target_features, dict) and len(data.target_features) > 0:
            print(f"Sample feature keys: {list(data.target_features.keys())[:5]}")
    
    # Update session data
    session_data["total_questions"] += 1

    is_correct = data.correct

    if is_correct:
        session_data["correct_answers"] += 1

    # Initialize feature stats if not present
    if "feature_stats" not in session_data:
        session_data["feature_stats"] = {}
        print("Initialized empty feature_stats in session_data")

    # Get database session
    db = next(get_db())

    # Get or create user stats record
    user_stats = db.query(models.GameStats).filter(models.GameStats.user_id == current_user.id).first()
    if not user_stats:
        print(f"Creating new GameStats record for user {current_user.id}")
        user_stats = models.GameStats(
            user_id=current_user.id,
            total_questions=0,
            correct_answers=0,
            feature_stats={}
        )
        db.add(user_stats)

    # Update database stats
    user_stats.total_questions += 1
    if is_correct:
        user_stats.correct_answers += 1

    # Initialize feature stats dictionary if needed
    if not user_stats.feature_stats:
        user_stats.feature_stats = {}
        print("Initialized empty feature_stats in database")
    elif isinstance(user_stats.feature_stats, str):
        # Handle case where feature_stats might be stored as a string
        try:
            user_stats.feature_stats = json.loads(user_stats.feature_stats)
            print("Parsed feature_stats from JSON string")
        except:
            print("Failed to parse feature_stats string, resetting to empty dict")
            user_stats.feature_stats = {}

    # Record the attempt
    attempt_data = {
        "shape_id": data.shape_id,
        "correct": is_correct,
        "error_type": data.error_type,
        "response_time": data.response_time,
        "target_id": data.target_id,
        "user_id": current_user.id,
        "task_variant": data.task_variant,
        "mirror_mode": data.mirror_mode,
        "interaction_data": data.interaction_data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    session_data["attempts"].append(attempt_data)

    # Process target features if available
    print(f"Target features received: {data.target_features is not None}")
    if data.target_features:
        try:
            # Get features dictionary
            features_dict = {}
            
            # Handle various feature format possibilities
            if isinstance(data.target_features, dict):
                features_dict = data.target_features
                print(f"Features received as dictionary with {len(features_dict)} keys")
                if len(features_dict) > 0:
                    print(f"First 5 feature keys: {list(features_dict.keys())[:5]}")
            elif hasattr(data.target_features, "dict"):
                features_dict = data.target_features.to_dict()
                print("Features received as Pydantic model")
            elif hasattr(data.target_features, "model_dump"):
                features_dict = data.target_features.model_dump()
                print("Features received with model_dump method")
            else:
                # Try to convert from JSON string if it's not already a dict
                if isinstance(data.target_features, str):
                    try:
                        features_dict = json.loads(data.target_features)
                        print("Features parsed from JSON string")
                    except json.JSONDecodeError:
                        print(f"Failed to parse features string: {data.target_features[:100]}...")
                        features_dict = {}
                else:
                    print(f"Features received as unknown type: {type(data.target_features)}")
                    features_dict = data.target_features

            print(f"Features dict type: {type(features_dict)}")
            print(f"Feature keys: {list(features_dict.keys()) if isinstance(features_dict, dict) else 'Not a dict'}")
            print(f"Features received: {len(features_dict) if isinstance(features_dict, dict) else 0}")

            # Add detailed debug info
            if isinstance(features_dict, dict) and len(features_dict) > 0:
                print(f"Sample feature keys: {list(features_dict.keys())[:5]}")
                print(f"Sample feature value types: {[type(features_dict[k]) for k in list(features_dict.keys())[:3]]}")
                
                # Check sample feature values to verify they're valid
                for i, key in enumerate(list(features_dict.keys())[:3]):
                    value = features_dict[key]
                    print(f"Sample feature {i+1}: {key} = {value} (type: {type(value)})")
            else:
                print("WARNING: Empty or invalid features dictionary received")
                # If features_dict is empty or invalid, try to extract it from ShapeFeatureSet model fields
                if hasattr(ShapeFeatureSet, "model_fields"):
                    print("Attempting to use default feature set from ShapeFeatureSet model")
                    dummy_features = ShapeFeatureSet(
                        voxel_count=5,
                        grid_size=(7, 7, 7)
                    )
                    features_dict = dummy_features.to_dict()
                    print(f"Created dummy feature set with {len(features_dict)} fields")

            print(f"Processing features for response. Feature count: {len(features_dict)}")

            # Initialize feature_stats in user_stats if it's None or not a dict
            if not user_stats.feature_stats or not isinstance(user_stats.feature_stats, dict):
                user_stats.feature_stats = {}
                print("Initializing empty feature_stats dictionary in database")

            # Track features processed successfully
            processed_count = 0
            
            # Process each feature
            for field, value in features_dict.items():
                # Skip None values and grid_size tuples
                if value is None or (field == "grid_size" and isinstance(value, tuple)):
                    continue

                # Normalize value to string for dictionary keys
                try:
                    if isinstance(value, (list, tuple, dict)):
                        value_key = str(value)
                    elif isinstance(value, (int, float, str, bool)):
                        value_key = str(value)
                    else:
                        print(f"Skipping non-serializable value: {field}={type(value)}")
                        continue
                except Exception as e:
                    print(f"Error serializing value for {field}: {str(e)}")
                    continue

                try:
                    # Update session stats
                    if field not in session_data["feature_stats"]:
                        session_data["feature_stats"][field] = {}

                    if value_key not in session_data["feature_stats"][field]:
                        session_data["feature_stats"][field][value_key] = {
                            "correct": 0,
                            "incorrect": 0,
                            "response_times": []
                        }

                    session_stat = session_data["feature_stats"][field][value_key]
                    if is_correct:
                        session_stat["correct"] += 1
                        session_stat["response_times"].append(data.response_time)
                    else:
                        session_stat["incorrect"] += 1

                    # Ensure feature exists in database stats (with proper initialization)
                    if field not in user_stats.feature_stats:
                        print(f"Adding new feature '{field}' to database stats")
                        user_stats.feature_stats[field] = {}

                    if value_key not in user_stats.feature_stats[field]:
                        print(f"Adding new value '{value_key}' for feature '{field}' to database stats")
                        user_stats.feature_stats[field][value_key] = {
                            "correct": 0,
                            "incorrect": 0,
                            "response_times": []
                        }

                    # Update database stats with proper type checking
                    db_stat = user_stats.feature_stats[field][value_key]

                    # Convert db_stat to a dict if it's not already one
                    if not isinstance(db_stat, dict):
                        print(f"Fixing non-dict db_stat for {field}:{value_key}")
                        db_stat = {"correct": 0, "incorrect": 0, "response_times": []}
                        user_stats.feature_stats[field][value_key] = db_stat

                    if is_correct:
                        # Ensure correct is an integer
                        if "correct" not in db_stat:
                            db_stat["correct"] = 0
                        db_stat["correct"] = int(db_stat["correct"]) + 1

                        # Ensure response_times is a list
                        if "response_times" not in db_stat or not isinstance(db_stat["response_times"], list):
                            db_stat["response_times"] = []
                        db_stat["response_times"].append(data.response_time)
                    else:
                        # Ensure incorrect is an integer
                        if "incorrect" not in db_stat:
                            db_stat["incorrect"] = 0
                        db_stat["incorrect"] = int(db_stat["incorrect"]) + 1
                    
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing feature {field}: {str(e)}")

            print(f"Successfully processed {processed_count} features")
            print(f"Updated feature stats. Session features: {len(session_data['feature_stats'])}, DB features: {len(user_stats.feature_stats)}")
            print(f"Sample feature data in DB: {list(user_stats.feature_stats.keys())[:3] if user_stats.feature_stats else 'None'}")

            # More detailed diagnostic information
            if not user_stats.feature_stats:
                print("WARNING: No feature stats saved to database!")
            elif len(user_stats.feature_stats) == 0:
                print("WARNING: Empty feature stats dictionary saved to database!")
            else:
                # Show a sample of what got saved
                sample_feature = list(user_stats.feature_stats.keys())[0]
                sample_value = list(user_stats.feature_stats[sample_feature].keys())[0] if user_stats.feature_stats[sample_feature] else None
                print(f"Sample saved feature data: {sample_feature} -> {sample_value}")
                if sample_value:
                    print(f"Value structure: {user_stats.feature_stats[sample_feature][sample_value]}")
        except Exception as e:
            print(f"Error processing features: {str(e)}")
            import traceback
            traceback.print_exc()

    # Process interaction metrics as synthetic features (Layer 3 behavioral metrics)
    if data.interaction_data and isinstance(data.interaction_data, dict):
        try:
            summary = data.interaction_data.get("summary", {})
            angular_disparity = data.interaction_data.get("angular_disparity")
            interaction_features = {}

            # Angular disparity bucket (Shepard & Metzler analysis)
            if angular_disparity is not None:
                for threshold, label in [(30, "0-30"), (60, "30-60"), (90, "60-90"),
                                          (120, "90-120"), (150, "120-150")]:
                    if angular_disparity < threshold:
                        interaction_features["angular_disparity_bucket"] = label
                        break
                else:
                    interaction_features["angular_disparity_bucket"] = "150-180"

            # Interaction strategy classification
            total_ms = summary.get("total_interaction_time_ms", 0)
            rt_ms = data.response_time * 1000
            if rt_ms > 0:
                ratio = total_ms / rt_ms
                if ratio < 0.1:
                    interaction_features["interaction_strategy"] = "mental_rotation"
                elif ratio < 0.4:
                    interaction_features["interaction_strategy"] = "mixed"
                else:
                    interaction_features["interaction_strategy"] = "physical_rotation"

            # Process through same feature_stats pipeline
            for field, value_key in interaction_features.items():
                # Session stats
                if field not in session_data["feature_stats"]:
                    session_data["feature_stats"][field] = {}
                if value_key not in session_data["feature_stats"][field]:
                    session_data["feature_stats"][field][value_key] = {
                        "correct": 0, "incorrect": 0, "response_times": []
                    }
                stat = session_data["feature_stats"][field][value_key]
                if is_correct:
                    stat["correct"] += 1
                    stat["response_times"].append(data.response_time)
                else:
                    stat["incorrect"] += 1

                # Database stats
                if field not in user_stats.feature_stats:
                    user_stats.feature_stats[field] = {}
                if value_key not in user_stats.feature_stats[field]:
                    user_stats.feature_stats[field][value_key] = {
                        "correct": 0, "incorrect": 0, "response_times": []
                    }
                db_stat = user_stats.feature_stats[field][value_key]
                if is_correct:
                    db_stat["correct"] = int(db_stat.get("correct", 0)) + 1
                    if not isinstance(db_stat.get("response_times"), list):
                        db_stat["response_times"] = []
                    db_stat["response_times"].append(data.response_time)
                else:
                    db_stat["incorrect"] = int(db_stat.get("incorrect", 0)) + 1

            print(f"Interaction metrics: strategy={interaction_features.get('interaction_strategy')}, "
                  f"disparity_bucket={interaction_features.get('angular_disparity_bucket')}")
        except Exception as e:
            print(f"Error processing interaction metrics: {e}")

    # Ensure feature_stats is properly serializable before committing
    try:
        # Print debug info about current state
        if isinstance(user_stats.feature_stats, dict):
            feature_count = len(user_stats.feature_stats)
            entry_count = sum(len(values) for values in user_stats.feature_stats.values() if isinstance(values, dict))
            print(f"Feature stats before serialization: {feature_count} features, {entry_count} total entries")
        elif isinstance(user_stats.feature_stats, str):
            print(f"Feature stats is already a string of length {len(user_stats.feature_stats)}")
        else:
            print(f"Feature stats is of type {type(user_stats.feature_stats)}")

        # Check if feature_stats is already a string (previously serialized JSON)
        if isinstance(user_stats.feature_stats, str):
            # Deserialize to work with it
            try:
                user_stats.feature_stats = json.loads(user_stats.feature_stats)
                print("Converted string feature_stats to dictionary for processing")
            except:
                # If it's invalid JSON, reset it
                user_stats.feature_stats = {}
                print("Reset invalid feature_stats string to empty dict")

        # Test JSON serialization to verify data is valid
        json_str = json.dumps(user_stats.feature_stats)

        # Store as string in the database to avoid SQLAlchemy serialization issues
        user_stats.feature_stats = json_str
        print(f"Serialized feature stats to JSON string. Length: {len(json_str)}, Features: {len(json.loads(json_str))}")

    except Exception as e:
        print(f"WARNING: Feature stats not serializable: {str(e)}")
        # Create a cleaned version
        clean_stats = {}
        try:
            if isinstance(user_stats.feature_stats, dict):
                for feature, values in user_stats.feature_stats.items():
                    if isinstance(values, dict):
                        clean_stats[feature] = {}
                        for value_key, stats in values.items():
                            if isinstance(stats, dict):
                                clean_stats[feature][value_key] = {
                                    "correct": stats.get("correct", 0),
                                    "incorrect": stats.get("incorrect", 0),
                                    "response_times": [float(t) for t in stats.get("response_times", []) if isinstance(t, (int, float))]
                                }
            
            # Store as JSON string
            json_str = json.dumps(clean_stats)
            user_stats.feature_stats = json_str
            print(f"Cleaned and serialized feature stats. Features: {len(clean_stats)}")
        except Exception as e2:
            print(f"ERROR: Even clean serialization failed: {str(e2)}")
            # Ultimate fallback - empty dict as string
            user_stats.feature_stats = "{}"
            print("RESET: Feature stats reset to empty JSON object")

    # Commit changes to the database
    try:
        db.commit()
        print(f"Successfully saved game stats to database. User ID: {current_user.id}, Total questions: {user_stats.total_questions}")
        print(f"Feature stats data size: {len(user_stats.feature_stats) if user_stats.feature_stats else 0} characters")
        
        # Verify stored data can be parsed
        try:
            if isinstance(user_stats.feature_stats, str):
                test_parse = json.loads(user_stats.feature_stats)
                print(f"Verification: Stored JSON can be parsed into {len(test_parse)} features")
        except Exception as e:
            print(f"WARNING: Stored JSON verification failed: {str(e)}")
    except Exception as e:
        db.rollback()
        print(f"Error saving to database: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

    return {
        "status": "recorded",
        "score": session_data["correct_answers"],
        "total": session_data["total_questions"],
        "features_processed": True
    }

@app.get("/get-session-stats")
def get_session_stats(current_user: models.User = Depends(auth.get_current_active_user)):
    return session_data

def _compute_scorecard(user_id: int) -> dict:
    """Compute scorecard data for a user. Reusable by both /scorecard and coaching endpoints."""
    # Process session data
    session_full_stats = {}
    session_feature_stats = session_data.get("feature_stats", {})

    for feature in ShapeFeatureSet.model_fields:
        session_full_stats[feature] = {}
        if feature in session_feature_stats:
            for value, record in session_feature_stats[feature].items():
                total_attempts = record["correct"] + record["incorrect"]
                avg_time = (sum(record["response_times"]) / len(record["response_times"])
                            if record["response_times"] else None)

                # Calculate success rate percentage
                success_rate = (record["correct"] / total_attempts * 100) if total_attempts > 0 else 0

                session_full_stats[feature][value] = {
                    "correct": record["correct"],
                    "incorrect": record["incorrect"],
                    "total_attempts": total_attempts,
                    "success_rate": round(success_rate, 2),
                    "avg_response_time": avg_time
                }

    # Get database session
    db = next(get_db())

    # Get user stats from database
    user_stats = db.query(models.GameStats).filter(models.GameStats.user_id == user_id).first()

    # Create default stats if none exist
    if not user_stats:
        user_stats = models.GameStats(
            user_id=user_id,
            total_questions=0,
            correct_answers=0,
            feature_stats={}
        )
        db.add(user_stats)
        db.commit()

    # Process database stats
    persistent_full_stats = {}

    # Handle feature_stats that might be stored as a JSON string
    if isinstance(user_stats.feature_stats, str):
        try:
            persistent_feature_stats = json.loads(user_stats.feature_stats)
            print("Deserialized feature_stats from JSON string in scorecard")
        except:
            print("Failed to deserialize feature_stats string in scorecard")
            persistent_feature_stats = {}
    else:
        persistent_feature_stats = user_stats.feature_stats or {}

    for feature in ShapeFeatureSet.model_fields:
        persistent_full_stats[feature] = {}
        if feature in persistent_feature_stats:
            for value, record in persistent_feature_stats[feature].items():
                total_attempts = record.get("correct", 0) + record.get("incorrect", 0)
                response_times = record.get("response_times", [])
                avg_time = (sum(response_times) / len(response_times)
                            if response_times else None)

                # Calculate success rate percentage
                success_rate = (record.get("correct", 0) / total_attempts * 100) if total_attempts > 0 else 0

                persistent_full_stats[feature][value] = {
                    "correct": record.get("correct", 0),
                    "incorrect": record.get("incorrect", 0),
                    "total_attempts": total_attempts,
                    "success_rate": round(success_rate, 2),
                    "avg_response_time": avg_time
                }

    # Calculate overall performance metrics
    session_total = session_data["total_questions"]
    session_correct = session_data["correct_answers"]

    persistent_total = user_stats.total_questions
    persistent_correct = user_stats.correct_answers

    # Find weak areas for session (features with < 50% success rate and at least 3 attempts)
    session_weak_areas = []
    for feature, values in session_full_stats.items():
        for value, stats in values.items():
            if stats["total_attempts"] >= 3 and stats["success_rate"] < 50:
                session_weak_areas.append({
                    "feature": feature,
                    "value": value,
                    "success_rate": stats["success_rate"],
                    "attempts": stats["total_attempts"]
                })

    # Sort weak areas by success rate (ascending)
    session_weak_areas.sort(key=lambda x: x["success_rate"])

    # Find weak areas for persistent data
    persistent_weak_areas = []
    for feature, values in persistent_full_stats.items():
        for value, stats in values.items():
            if stats["total_attempts"] >= 3 and stats["success_rate"] < 50:
                persistent_weak_areas.append({
                    "feature": feature,
                    "value": value,
                    "success_rate": stats["success_rate"],
                    "attempts": stats["total_attempts"]
                })

    # Sort weak areas by success rate (ascending)
    persistent_weak_areas.sort(key=lambda x: x["success_rate"])

    # Add diagnostic information
    print(f"Session data - Total questions: {session_total}, Features with data: {len(session_data.get('feature_stats', {}))}")
    print(f"Database data - Total questions: {persistent_total}, Features with data: {len(persistent_feature_stats)}")

    if persistent_total > 0 and not persistent_feature_stats:
        print("WARNING: Database data has questions but no feature stats!")

    db.close()

    # Compute per-task-variant breakdown from session attempts
    task_variant_stats = {}
    for attempt in session_data.get("attempts", []):
        variant = attempt.get("task_variant", "simultaneous") or "simultaneous"
        mirror = attempt.get("mirror_mode", False)
        key = f"{variant}{'_mirror' if mirror else ''}"
        if key not in task_variant_stats:
            task_variant_stats[key] = {"total": 0, "correct": 0}
        task_variant_stats[key]["total"] += 1
        if attempt.get("correct"):
            task_variant_stats[key]["correct"] += 1
    for key, stats in task_variant_stats.items():
        stats["accuracy"] = round((stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0, 2)

    # Compute interaction scorecard (Layer 3 behavioral metrics)
    INTERACTION_FEATURES = ["angular_disparity_bucket", "interaction_strategy"]

    def _extract_interaction_stats(feature_source):
        stats = {}
        for feature in INTERACTION_FEATURES:
            stats[feature] = {}
            if feature in feature_source:
                for value, record in feature_source[feature].items():
                    total = record.get("correct", 0) + record.get("incorrect", 0)
                    rts = record.get("response_times", [])
                    avg_rt = (sum(rts) / len(rts)) if rts else None
                    sr = (record.get("correct", 0) / total * 100) if total > 0 else 0
                    stats[feature][value] = {
                        "correct": record.get("correct", 0),
                        "incorrect": record.get("incorrect", 0),
                        "total_attempts": total,
                        "success_rate": round(sr, 2),
                        "avg_response_time": avg_rt,
                    }
        return stats

    session_interaction = _extract_interaction_stats(session_feature_stats)
    persistent_interaction = _extract_interaction_stats(persistent_feature_stats)

    # Collect raw RT-by-angular-disparity data for Shepard & Metzler slope analysis
    rt_by_disparity = []
    for attempt in session_data.get("attempts", []):
        idata = attempt.get("interaction_data")
        if idata and idata.get("angular_disparity") is not None:
            rt_by_disparity.append({
                "angular_disparity": idata["angular_disparity"],
                "response_time": attempt["response_time"],
                "correct": attempt.get("correct", False),
            })

    # Collect recent raw attempts with full context for LLM cross-dimensional analysis
    recent_attempts_full = []
    for attempt in session_data.get("attempts", [])[-15:]:
        entry = {
            "correct": attempt.get("correct"),
            "response_time": attempt.get("response_time"),
            "task_variant": attempt.get("task_variant"),
            "mirror_mode": attempt.get("mirror_mode"),
        }
        idata = attempt.get("interaction_data")
        if idata:
            entry["angular_disparity"] = idata.get("angular_disparity")
            summary = idata.get("summary", {})
            entry["rotation_events"] = summary.get("total_rotation_events", 0)
            entry["interaction_time_ms"] = summary.get("total_interaction_time_ms", 0)
        recent_attempts_full.append(entry)

    return {
        "session": {
            "total_questions": session_total,
            "correct_answers": session_correct,
            "accuracy": round((session_correct / session_total * 100) if session_total > 0 else 0, 2),
            "feature_scorecard": session_full_stats,
            "weak_areas": session_weak_areas[:5],  # Return top 5 weakest areas
            "task_variant_stats": task_variant_stats,
            "interaction_scorecard": session_interaction,
            "rt_by_angular_disparity": rt_by_disparity,
            "recent_attempts_full": recent_attempts_full,
            "debug_info": {
                "features_count": len(session_data.get('feature_stats', {})),
                "has_feature_data": bool(session_data.get('feature_stats'))
            }
        },
        "cumulative": {
            "total_questions": persistent_total,
            "correct_answers": persistent_correct,
            "accuracy": round((persistent_correct / persistent_total * 100) if persistent_total > 0 else 0, 2),
            "feature_scorecard": persistent_full_stats,
            "weak_areas": persistent_weak_areas[:5],  # Return top 5 weakest areas
            "interaction_scorecard": persistent_interaction,
            "debug_info": {
                "features_count": len(persistent_feature_stats),
                "has_feature_data": bool(persistent_feature_stats),
                "data_source": "database"
            }
        }
    }


@app.get("/scorecard")
def get_scorecard(current_user: models.User = Depends(auth.get_current_active_user)):
    return _compute_scorecard(current_user.id)


# ---------------------------------------------------------------------------
# LLM Coaching Endpoints
# ---------------------------------------------------------------------------

@app.get("/coaching/status")
def coaching_status():
    """Check if the LLM coaching service is available."""
    from llm_coaching import check_llm_available, LLM_MODEL, LLM_BASE_URL
    available = check_llm_available()
    return {
        "available": available,
        "model": LLM_MODEL if available else None,
        "endpoint": LLM_BASE_URL,
    }


@app.post("/coaching/feedback", response_model=CoachingResponse)
def coaching_feedback(
    request: CoachingFeedbackRequest,
    current_user: models.User = Depends(auth.get_current_active_user),
):
    """Quick feedback after a response (1-3 sentences)."""
    import time as _time
    from llm_coaching import (
        get_coaching_response, build_quick_feedback_prompt,
        get_fallback_coaching, LLM_MODEL,
    )

    start = _time.time()

    total_q = session_data["total_questions"]
    correct_q = session_data["correct_answers"]
    accuracy = round(correct_q / max(1, total_q) * 100)
    session_summary = f"Session: {total_q} questions, {correct_q} correct ({accuracy}%)"

    prompt = build_quick_feedback_prompt(
        was_correct=request.was_correct,
        response_time=request.response_time,
        target_features=request.target_features or {},
        recent_attempts=session_data.get("attempts", [])[-10:],
        session_stats_summary=session_summary,
        interaction_data=request.interaction_data,
    )

    response_text = get_coaching_response(prompt, max_tokens=150)
    latency = (_time.time() - start) * 1000

    if response_text:
        return CoachingResponse(
            coaching_text=response_text,
            source="llm",
            model=LLM_MODEL,
            latency_ms=round(latency, 1),
        )

    # Fallback
    weak = []
    fs = session_data.get("feature_stats", {})
    for feature, values in fs.items():
        for value, stats in values.items():
            total = stats.get("correct", 0) + stats.get("incorrect", 0)
            if total >= 3:
                rate = round(stats["correct"] / total * 100, 1)
                if rate < 50:
                    weak.append({"feature": feature, "success_rate": rate})
    weak.sort(key=lambda x: x["success_rate"])

    return CoachingResponse(
        coaching_text=get_fallback_coaching(weak),
        source="fallback",
        latency_ms=round(latency, 1),
    )


@app.post("/coaching/session-summary", response_model=CoachingResponse)
def coaching_session_summary(
    current_user: models.User = Depends(auth.get_current_active_user),
):
    """Comprehensive analysis at end of session."""
    import time as _time
    from llm_coaching import (
        get_coaching_response, build_session_summary_prompt,
        serialize_performance_for_llm, get_fallback_coaching, LLM_MODEL,
    )

    start = _time.time()

    scorecard = _compute_scorecard(current_user.id)
    performance_summary = serialize_performance_for_llm(scorecard)

    prompt = build_session_summary_prompt(performance_summary)
    response_text = get_coaching_response(prompt, max_tokens=600)
    latency = (_time.time() - start) * 1000

    if response_text:
        return CoachingResponse(
            coaching_text=response_text,
            source="llm",
            model=LLM_MODEL,
            latency_ms=round(latency, 1),
        )

    weak = scorecard.get("session", {}).get("weak_areas", [])
    return CoachingResponse(
        coaching_text=get_fallback_coaching(weak),
        source="fallback",
        latency_ms=round(latency, 1),
    )


@app.post("/coaching/advice", response_model=CoachingResponse)
def coaching_advice(
    request: CoachingAdviceRequest,
    current_user: models.User = Depends(auth.get_current_active_user),
):
    """On-demand coaching advice (user-initiated or scorecard page)."""
    import time as _time
    from llm_coaching import (
        get_coaching_response, build_on_demand_advice_prompt,
        build_scorecard_analysis_prompt, serialize_performance_for_llm,
        get_fallback_coaching, LLM_MODEL,
    )

    start = _time.time()

    scorecard = _compute_scorecard(current_user.id)
    performance_summary = serialize_performance_for_llm(scorecard)

    # Use the detailed scorecard prompt if no user question (likely scorecard page)
    if request.user_question:
        prompt = build_on_demand_advice_prompt(performance_summary, request.user_question)
        max_tok = 400
    else:
        prompt = build_on_demand_advice_prompt(performance_summary)
        max_tok = 400

    response_text = get_coaching_response(prompt, max_tokens=max_tok)
    latency = (_time.time() - start) * 1000

    if response_text:
        return CoachingResponse(
            coaching_text=response_text,
            source="llm",
            model=LLM_MODEL,
            latency_ms=round(latency, 1),
        )

    weak = scorecard.get("session", {}).get("weak_areas", [])
    return CoachingResponse(
        coaching_text=get_fallback_coaching(weak),
        source="fallback",
        latency_ms=round(latency, 1),
    )


@app.post("/coaching/scorecard-analysis", response_model=CoachingResponse)
def coaching_scorecard_analysis(
    current_user: models.User = Depends(auth.get_current_active_user),
):
    """Detailed cognitive analysis for the scorecard page."""
    import time as _time
    from llm_coaching import (
        get_coaching_response, build_scorecard_analysis_prompt,
        serialize_performance_for_llm, get_fallback_coaching, LLM_MODEL,
    )

    start = _time.time()

    scorecard = _compute_scorecard(current_user.id)
    performance_summary = serialize_performance_for_llm(scorecard)

    prompt = build_scorecard_analysis_prompt(performance_summary)
    response_text = get_coaching_response(prompt, max_tokens=800)
    latency = (_time.time() - start) * 1000

    if response_text:
        return CoachingResponse(
            coaching_text=response_text,
            source="llm",
            model=LLM_MODEL,
            latency_ms=round(latency, 1),
        )

    weak = scorecard.get("session", {}).get("weak_areas", [])
    return CoachingResponse(
        coaching_text=get_fallback_coaching(weak),
        source="fallback",
        latency_ms=round(latency, 1),
    )


@app.post("/reset-session")
def reset_session(current_user: models.User = Depends(auth.get_current_active_user)):
    session_data["total_questions"] = 0
    session_data["correct_answers"] = 0
    session_data["attempts"] = []
    session_data["feature_errors"] = {}
    session_data["feature_stats"] = {}
    return {"message": "Session reset successfully"}

@app.post("/reset-all-data")
def reset_all_data(current_user: models.User = Depends(auth.get_current_active_user)):
    # Reset session data
    session_data["total_questions"] = 0
    session_data["correct_answers"] = 0
    session_data["attempts"] = []
    session_data["feature_errors"] = {}
    session_data["feature_stats"] = {}

    # Reset database data
    db = next(get_db())
    user_stats = db.query(models.GameStats).filter(models.GameStats.user_id == current_user.id).first()

    if user_stats:
        user_stats.total_questions = 0
        user_stats.correct_answers = 0
        user_stats.feature_stats = {}
        db.commit()

    db.close()

    return {"message": "All data reset successfully"}

@app.get("/profile")
def get_profile(current_user: models.User = Depends(auth.get_current_active_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at
    }

@app.post("/repair-feature-data")
def repair_feature_data(current_user: models.User = Depends(auth.get_current_active_user)):
    """
    Attempt to repair feature data by:
    1. Retrieving any stored feature data from session and database
    2. Consolidating and repairing any inconsistencies
    3. Storing the repaired data back to the database
    """
    print(f"\n=== Attempting Feature Data Repair ===")
    print(f"User: {current_user.username} (ID: {current_user.id})")

    # Get database session
    db = next(get_db())
    
    try:
        # 1. Get current user stats record from database
        user_stats = db.query(models.GameStats).filter(models.GameStats.user_id == current_user.id).first()
        
        if not user_stats:
            print("No user stats found in database, creating new record")
            user_stats = models.GameStats(
                user_id=current_user.id,
                total_questions=0,
                correct_answers=0,
                feature_stats={}
            )
            db.add(user_stats)
            db.commit()
            return {"message": "No data to repair. New stats record created."}
            
        # 2. Ensure proper deserialization of feature_stats
        db_feature_stats = {}
        if user_stats.feature_stats:
            if isinstance(user_stats.feature_stats, str):
                try:
                    db_feature_stats = json.loads(user_stats.feature_stats)
                    print(f"Parsed feature_stats from JSON string, found {len(db_feature_stats)} features")
                except json.JSONDecodeError:
                    print("Failed to parse feature_stats string")
                    db_feature_stats = {}
            elif isinstance(user_stats.feature_stats, dict):
                db_feature_stats = user_stats.feature_stats
                print(f"Found dictionary feature_stats with {len(db_feature_stats)} features")
            else:
                print(f"Unknown feature_stats type: {type(user_stats.feature_stats)}")
                db_feature_stats = {}
        
        # 3. Get session feature stats
        session_feature_stats = session_data.get("feature_stats", {})
        print(f"Session feature_stats has {len(session_feature_stats)} features")
        
        # 4. Create dummy feature data if both sources are empty but we have questions
        if not db_feature_stats and not session_feature_stats and user_stats.total_questions > 0:
            print(f"Both feature stats empty but have {user_stats.total_questions} questions, creating dummy data")
            
            # Create basic data from ShapeFeatureSet fields
            if hasattr(ShapeFeatureSet, "model_fields"):
                dummy_features = ShapeFeatureSet(
                    voxel_count=5,
                    grid_size=(7, 7, 7)
                )
                dummy_fields = dummy_features.to_dict()
                
                # For each field, create a sample record
                feature_count = 0
                for field, value in dummy_fields.items():
                    # Skip complex fields and just use simple ones
                    if isinstance(value, (int, float, str)):
                        if field not in db_feature_stats:
                            db_feature_stats[field] = {}
                        
                        # Create dummy entry with 50% success rate
                        value_key = str(value)
                        if value_key not in db_feature_stats[field]:
                            correct_count = max(1, int(user_stats.correct_answers / 2))
                            incorrect_count = max(1, user_stats.total_questions - correct_count)
                            
                            db_feature_stats[field][value_key] = {
                                "correct": correct_count,
                                "incorrect": incorrect_count,
                                "response_times": [1.0]
                            }
                            feature_count += 1
                
                print(f"Created {feature_count} dummy feature records")
        
        # 5. Merge session data with database data
        combined_stats = copy.deepcopy(db_feature_stats)
        for feature, feature_values in session_feature_stats.items():
            if feature not in combined_stats:
                combined_stats[feature] = {}
                
            for value_key, stats in feature_values.items():
                if value_key not in combined_stats[feature]:
                    combined_stats[feature][value_key] = stats
                else:
                    # Merge the stats
                    db_stats = combined_stats[feature][value_key]
                    db_stats["correct"] = (db_stats.get("correct", 0) + stats.get("correct", 0))
                    db_stats["incorrect"] = (db_stats.get("incorrect", 0) + stats.get("incorrect", 0))
                    
                    # Merge response times
                    if "response_times" not in db_stats:
                        db_stats["response_times"] = []
                    db_stats["response_times"].extend(stats.get("response_times", []))
        
        # 6. Clean up the combined data for storage
        cleaned_stats = {}
        for feature, values in combined_stats.items():
            if not values:  # Skip empty values
                continue
                
            cleaned_stats[feature] = {}
            for value_key, stats in values.items():
                if not stats:  # Skip empty stats
                    continue
                    
                try:
                    # Ensure the stats have the right format
                    clean_stat = {
                        "correct": int(stats.get("correct", 0)),
                        "incorrect": int(stats.get("incorrect", 0)),
                        "response_times": [float(t) for t in stats.get("response_times", []) if isinstance(t, (int, float))]
                    }
                    
                    # Only include if there's actual data
                    if clean_stat["correct"] > 0 or clean_stat["incorrect"] > 0:
                        cleaned_stats[feature][value_key] = clean_stat
                except Exception as e:
                    print(f"Error cleaning stat {feature}:{value_key}: {str(e)}")
        
        # 7. Store back to the database
        try:
            json_str = json.dumps(cleaned_stats)
            user_stats.feature_stats = json_str
            db.commit()
            print(f"Repaired and saved feature stats: {len(cleaned_stats)} features")
            
            # Also update session data for consistency
            session_data["feature_stats"] = cleaned_stats
            
            result = {
                "message": f"Data repair complete. Saved {len(cleaned_stats)} features with data.",
                "features_repaired": len(cleaned_stats)
            }
        except Exception as e:
            db.rollback()
            print(f"Error saving repaired data: {str(e)}")
            result = {"message": f"Error saving repaired data: {str(e)}"}
    
    except Exception as e:
        print(f"Error during feature data repair: {str(e)}")
        import traceback
        traceback.print_exc()
        result = {"message": f"Error during repair: {str(e)}"}
    finally:
        db.close()
    
    return result


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    # Re-enable reload for development convenience
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)