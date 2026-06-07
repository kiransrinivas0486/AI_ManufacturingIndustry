# app5.py
# Manufacturing Workflow AI - Clean Baseline Version

from openai import OpenAI
import streamlit as st
import cv2
import base64
import pandas as pd
import tempfile
import time

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

st.set_page_config(
    page_title="Manufacturing Workflow AI",
    layout="wide"
)


if "human_overrides" not in st.session_state:
    st.session_state.human_overrides = {}


st.title("🏭 Manufacturing Workflow Intelligence")

st.write(
    "Upload a machining workflow video for AI-powered process analysis."
)

uploaded_file = st.file_uploader(
    "Upload Manufacturing Video",
    type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📹 Video")
        st.video(uploaded_file)

    with col2:

        st.subheader("📊 Analysis Results")

        if st.button("Analyze Workflow"):

            with st.spinner("Analyzing workflow..."):

                temp_video = tempfile.NamedTemporaryFile(delete=False)
                temp_video.write(uploaded_file.read())
                video_path = temp_video.name

                cap = cv2.VideoCapture(video_path)

                fps = cap.get(cv2.CAP_PROP_FPS)

                st.write(f"🎥 FPS Detected: {fps:.2f}")

                frame_interval = int(fps * 2)
                max_frames = 8

                

                frame_count = 0
                processed_frames = 0

                raw_observations = []
                timestamp_buffer = []
                frame_buffer = [] 
                progress = st.progress(0)
                st.write(f"Clip Size: {len(frame_buffer)}")

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    if frame_count % frame_interval == 0:

                        timestamp = frame_count / fps

                        frame = cv2.resize(frame, (640, 360))

                        frame_filename = f"frame_{int(timestamp)}.jpg"

                        cv2.imwrite(frame_filename, frame)

                        with open(frame_filename, "rb") as image_file:

                            image_base64 = base64.b64encode(
                                image_file.read()
                            ).decode("utf-8")
                        frame_buffer.append(image_base64)
                        timestamp_buffer.append(timestamp)
                        try:
                            if len(frame_buffer) < 4:
                             frame_count += 1
                             continue
                            clip_start = timestamp_buffer[0]
                            clip_end = timestamp_buffer[-1]
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": """
You are a manufacturing workflow analyst.

Observe the manufacturing activity carefully. Do NOT classify immediately.

- Component Pickup
- Machine Loading
- Machine Running
- Part Removal
- Inspection / Measurement
- Green Marking
- Tray Placement
- Waiting / Idle

Machine Loading:
Operator is actively inserting, positioning, adjusting, loading, unloading,
or handling a component at the machine.

Machine Running:
Machine is operating and the operator is mainly monitoring, waiting,
observing, or not actively manipulating the component.

If multiple activities occur, choose the PRIMARY activity occupying most of the clip.

Return STRICTLY:

Observation: <what is happening>
Motion: <how objects/person move>
Confidence: High / Medium / Low
"""
                                    },
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": """You are receiving consecutive images from a machining operation.

Treat all images as one short manufacturing video clip. Use motion across the sequence. Return Step Name, Confidence and Evidence only."""
                                            },
                                         ]
                                         +
                                         [
                                        {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{img}"
                                                }
                                            }
                                            for img in frame_buffer                                        
                                        ]
                                    }
                                ]
                            )

                            analysis = response.choices[0].message.content
                            frame_buffer = []
                            timestamp_buffer = []
                            #Use below for debugging raw GPT response
                            st.write("RAW GPT RESPONSE")
                            st.write(analysis)
                        except Exception as e:

                            st.error(f"OpenAI Error: {e}")
                            break

                        step_name = "Waiting / Idle"
                        confidence = "Unknown"
                        evidence = ""

                        for line in analysis.split("\n"):

                            clean_line = line.replace("*", "").strip()

                            if "Step Name:" in clean_line:
                                step_name = clean_line.split("Step Name:")[1].strip()

                            elif "Confidence:" in clean_line:
                                confidence = clean_line.split("Confidence:")[1].strip()

                            elif "Evidence:" in clean_line:
                                evidence = clean_line.split("Evidence:")[1].strip()

                        raw_observations.append({
                            "timestamp": clip_start,
                            "clip_end": clip_end,
                            "step_name": step_name,
                            "confidence": confidence,
                            "evidence": evidence
                        })

                        processed_frames += 1

                        progress.progress(
                            min(processed_frames / max_frames, 1.0)
                        )

                        time.sleep(1)

                        if processed_frames >= max_frames:
                            break

                    frame_count += 1

                cap.release()

                
                # ==========================
                # HUMAN IN THE LOOP REVIEW
                # ==========================
                st.subheader("🧑 Human Review")

                if len(raw_observations) > 0:
                    review_rows = []
                    for obs in raw_observations:
                        review_rows.append({
                            "Start": int(obs["timestamp"]),
                            "End": int(obs["clip_end"]),
                            "GPT Step": obs["step_name"],
                            "Confidence": obs["confidence"]
                        })
                    st.dataframe(pd.DataFrame(review_rows), use_container_width=True)

                    st.info("Current version shows GPT detections for review. Manual override can be added in the next iteration.")

                STEP_OPTIONS = [
                    "Component Pickup",
                    "Machine Loading",
                    "Machine Running",
                    "Part Removal",
                    "Inspection / Measurement",
                    "Green Marking",
                    "Tray Placement",
                    "Waiting / Idle"
                ]

                st.write("Review all clips, then click Apply Human Corrections.")

                overrides = {}

                with st.form("human_review_form"):

                    for idx, obs in enumerate(raw_observations):

                        current_step = obs.get("step_name", "Waiting / Idle")
                        if current_step not in STEP_OPTIONS:
                            current_step = "Waiting / Idle"

                        overrides[idx] = st.selectbox(
                            f"Clip {idx}: {int(obs['timestamp'])}-{int(obs['clip_end'])} sec",
                            STEP_OPTIONS,
                            index=STEP_OPTIONS.index(current_step),
                            key=f"review_{idx}"
                        )

                    submitted = st.form_submit_button(
                        "✅ Apply Human Corrections"
                    )

                if submitted:

                    for idx, selected_step in overrides.items():

                        raw_observations[idx]["step_name"] = selected_step
                        raw_observations[idx]["confidence"] = "Human Validated"

                        clip_key = (
                            int(raw_observations[idx]["timestamp"]),
                            int(raw_observations[idx]["clip_end"])
                        )

                        st.session_state.human_overrides[clip_key] = selected_step

                    st.success("Human corrections applied and saved.")



                # Apply previously saved human corrections
                for obs in raw_observations:

                    clip_key = (
                        int(obs["timestamp"]),
                        int(obs["clip_end"])
                    )

                    if clip_key in st.session_state.human_overrides:

                        obs["step_name"] = st.session_state.human_overrides[clip_key]
                        obs["confidence"] = "Human Locked"


                workflow_results = []

                if len(raw_observations) > 0:

                    classified_steps = []
                    for obs in raw_observations:
                        classified_steps.append({
                            "timestamp": obs["timestamp"],
                            "clip_end": obs["clip_end"],
                            "step_name": obs["step_name"],
                            "confidence": obs["confidence"],
                            "analysis": obs.get("evidence","")
                        })

                    current_step = classified_steps[0]

                    start_time = current_step["timestamp"]
                    current_name = current_step["step_name"]
                    current_confidence = current_step["confidence"]
                    current_analysis = current_step["analysis"]

                    for i in range(1, len(classified_steps)):

                        next_step = classified_steps[i]

                        if next_step["step_name"] != current_name:

                            end_time = next_step["timestamp"]

                            workflow_results.append({
                                "Start Time": f"{int(start_time)} sec",
                                "End Time": f"{int(end_time)} sec",
                                "Duration": f"{int(end_time - start_time)} sec",
                                "Step Name": current_name,
                                "Confidence": current_confidence,
                                "Detailed Analysis": current_analysis
                            })

                            start_time = next_step["timestamp"]
                            current_name = next_step["step_name"]
                            current_confidence = next_step["confidence"]
                            current_analysis = next_step["analysis"]

                    final_timestamp = classified_steps[-1]["clip_end"]

                    workflow_results.append({
                        "Start Time": f"{int(start_time)} sec",
                        "End Time": f"{int(final_timestamp)} sec",
                        "Duration": f"{int(final_timestamp - start_time)} sec",
                        "Step Name": current_name,
                        "Confidence": current_confidence,
                        "Detailed Analysis": current_analysis
                    })

                machine_utilization = 0.0
                operator_utilization = 0.0

                if len(workflow_results) > 0:

                    def parse_sec(value):
                        return int(value.replace(" sec", ""))

                    total_observed_seconds = (
                        parse_sec(workflow_results[-1]["End Time"])
                        - parse_sec(workflow_results[0]["Start Time"])
                    )

                    machine_running_seconds = sum(
                        parse_sec(row["Duration"])
                        for row in workflow_results
                        if row["Step Name"] == "Machine Running"
                    )

                    operator_active_steps = {
                        "Component Pickup",
                        "Machine Loading",
                        "Part Removal",
                        "Inspection / Measurement",
                        "Green Marking",
                        "Tray Placement",
                    }

                    operator_active_seconds = sum(
                        parse_sec(row["Duration"])
                        for row in workflow_results
                        if row["Step Name"] in operator_active_steps
                    )

                    if total_observed_seconds > 0:
                        machine_utilization = (
                            machine_running_seconds / total_observed_seconds
                        ) * 100
                        operator_utilization = (
                            operator_active_seconds / total_observed_seconds
                        ) * 100

                # ==========================
                # WORKFLOW STATE VALIDATION
                # ==========================

                VALID_NEXT = {
                    "Component Pickup": ["Machine Loading"],
                    "Machine Loading": ["Machine Running"],
                    "Machine Running": ["Part Removal"],
                    "Part Removal": ["Inspection / Measurement"],
                    "Inspection / Measurement": ["Green Marking", "Tray Placement"],
                    "Green Marking": ["Tray Placement"],
                    "Tray Placement": ["Component Pickup"]
                }

                cleaned_results = []

                if len(workflow_results) > 0:

                    cleaned_results.append(workflow_results[0])

                    for i in range(1, len(workflow_results)):

                        previous_step = cleaned_results[-1]["Step Name"]
                        current_step = workflow_results[i]["Step Name"]

                        valid_steps = VALID_NEXT.get(previous_step, [])

                        if current_step in valid_steps:
                            cleaned_results.append(workflow_results[i])
                        else:
                            workflow_results[i]["Confidence"] = "Low"
                            cleaned_results.append(workflow_results[i])

                    workflow_results = cleaned_results


                
                # ==========================
                # WORKFLOW INFERENCE ENGINE
                # ==========================
                inferred_results = []

                for row in workflow_results:

                    inferred_results.append(row)

                    step = row["Step Name"]

                    if step == "Green Marking":
                        inferred_results.append({
                            "Start Time": row["End Time"],
                            "End Time": row["End Time"],
                            "Duration": "0 sec",
                            "Step Name": "Tray Placement",
                            "Confidence": "Inferred",
                            "Detailed Analysis":
                            "Inferred from workflow: Green Marking is usually followed by Tray Placement."
                        })

                    elif step == "Machine Loading":
                        inferred_results.append({
                            "Start Time": row["End Time"],
                            "End Time": row["End Time"],
                            "Duration": "0 sec",
                            "Step Name": "Machine Running",
                            "Confidence": "Inferred",
                            "Detailed Analysis":
                            "Inferred from workflow: Machine Loading is usually followed by Machine Running."
                        })

                    elif step == "Inspection / Measurement":
                        inferred_results.append({
                            "Start Time": row["End Time"],
                            "End Time": row["End Time"],
                            "Duration": "0 sec",
                            "Step Name": "Green Marking",
                            "Confidence": "Inferred",
                            "Detailed Analysis":
                            "Inferred from workflow: Inspection is usually followed by Green Marking."
                        })

                workflow_results = inferred_results


                expected_steps = [
                    "Component Pickup",
                    "Machine Loading",
                    "Machine Running",
                    "Part Removal",
                    "Inspection / Measurement",
                    "Green Marking",
                    "Tray Placement"
                ]

                detected_steps = [
                    x["Step Name"]
                    for x in workflow_results
                ]

                missing_steps = [
                    x for x in expected_steps
                    if x not in detected_steps
                ]

                compliance_score = (
                    (len(expected_steps) - len(missing_steps))
                    / len(expected_steps)
                ) * 100

                st.subheader("📊 Compliance Summary")

                st.metric(
                    "Compliance Score",
                    f"{compliance_score:.1f}%"
                )

                st.subheader("⚙️ Operational KPIs")

                kpi_col1, kpi_col2 = st.columns(2)

                kpi_col1.metric(
                    "Machine Utilization",
                    f"{machine_utilization:.1f}%"
                )
                kpi_col2.metric(
                    "Operator Utilization",
                    f"{operator_utilization:.1f}%"
                )

                if missing_steps:

                    st.warning("Missing Steps Detected")

                    for step in missing_steps:
                        st.write(f"❌ {step}")

                else:

                    st.success("✅ All workflow steps detected")

                st.subheader("📋 Workflow Timeline")

                df = pd.DataFrame(workflow_results)

                st.dataframe(
                    df,
                    use_container_width=True
                )
