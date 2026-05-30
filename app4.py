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

                st.write(f"🎥 FPS Detected: {fps}")

                frame_interval = int(fps * 5)
                frame_count = 0

                max_frames = 12
                processed_frames = 0

                raw_steps = []

                progress = st.progress(0)

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    if frame_count % frame_interval == 0:

                        timestamp = frame_count / fps

                        frame = cv2.resize(
                            frame,
                            (640, 360)
                        )

                        frame_filename = f"frame_{int(timestamp)}.jpg"

                        cv2.imwrite(
                            frame_filename,
                            frame
                        )

                        with open(frame_filename, "rb") as image_file:

                            image_base64 = base64.b64encode(
                                image_file.read()
                            ).decode("utf-8")
                    try:
                        response = client.chat.completions.create(

                            model="gpt-4o-mini",

                            messages=[
                                {
    "role": "system",
    "content": """
You are a manufacturing process analyst specializing in machining operations.

You are observing a worker performing a repetitive production workflow.

Expected workflow sequence:

1. Component Pickup
2. Machine Loading
3. Machining Process
4. Part Removal
5. Green Paint Application
6. Tray Placement

Important:

- The workflow repeats continuously.
- Use only the step names listed above.
- Do NOT invent new step names.
- Do NOT provide generic manufacturing descriptions.
- Focus on visible evidence in the image.

Identification Rules:

Component Pickup:
- Worker picking up component from tray or table
- Component in hand before machine interaction

Machine Loading:
- Worker placing component into machine
- Worker interacting with machine opening, chuck, fixture, or loading area

Machining Process:
- Component inside machine
- Machine operating
- Worker waiting or machine actively processing

Part Removal:
- Worker removing finished component from machine
- Component being extracted from machine

Green Paint Application:
- Worker holding component near green paint station
- Worker using brush, marker, cloth, sponge, spray, or paint container
- Any visible green color on component
- Any activity that appears to be marking or coating the component

Tray Placement:
- Worker placing completed component into tray, bin, or collection area

Return STRICTLY in this format:

Step Name: <step>

Confidence: High / Medium / Low

Detailed Analysis: <specific action visible in image>
"""
},
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "The image is one frame from a repeating machining workflow. Identify the current workflow step."
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{image_base64}"
                                            }
                                        }
                                    ]
                                }
                            ]
                        )

                        analysis = response.choices[0].message.content
                    except Exception as e:
                        st.error(f"OpenAI Error: {e}")
                        break
                        time.sleep(1)

                        step_name = "Unknown"
                        detailed_analysis = analysis

                        try:

                            lines = analysis.split("\n")

                            for line in lines:

                                if line.startswith("Step Name:"):
                                    step_name = line.replace(
                                        "Step Name:",
                                        ""
                                    ).strip()

                                elif line.startswith(
                                    "Detailed Analysis:"
                                ):
                                    detailed_analysis = line.replace(
                                        "Detailed Analysis:",
                                        ""
                                    ).strip()

                        except Exception:
                            pass

                        raw_steps.append({
                            "timestamp": timestamp,
                            "step_name": step_name,
                            "analysis": detailed_analysis
                        })

                        processed_frames += 1

                        progress.progress(
                            min(processed_frames / 100, 1.0)
                        )

                        if processed_frames >= max_frames:
                            break

                    frame_count += 1

                cap.release()

                workflow_results = []

                if len(raw_steps) > 0:

                    current_step = raw_steps[0]

                    start_time = current_step["timestamp"]
                    current_name = current_step["step_name"]
                    current_analysis = current_step["analysis"]

                    for i in range(1, len(raw_steps)):

                        next_step = raw_steps[i]

                        if next_step["step_name"] != current_name:

                            end_time = next_step["timestamp"]

                            duration = end_time - start_time

                            workflow_results.append({
                                "Start Time": f"{int(start_time)} sec",
                                "End Time": f"{int(end_time)} sec",
                                "Duration": f"{int(duration)} sec",
                                "Step Name": current_name,
                                "Detailed Analysis": current_analysis
                            })

                            start_time = next_step["timestamp"]
                            current_name = next_step["step_name"]
                            current_analysis = next_step["analysis"]

                    final_timestamp = raw_steps[-1]["timestamp"]
                    final_duration = (
                        final_timestamp - start_time
                    )

                    workflow_results.append({
                        "Start Time": f"{int(start_time)} sec",
                        "End Time": f"{int(final_timestamp)} sec",
                        "Duration": f"{int(final_duration)} sec",
                        "Step Name": current_name,
                        "Detailed Analysis": current_analysis
                    })

                expected_steps = [
                    "Component Pickup",
                    "Machine Loading",
                    "Machining Process",
                    "Part Removal",
                    "Green Paint Application",
                    "Tray Placement"
                ]

                detected_steps = [
                    step["Step Name"]
                    for step in workflow_results
                ]

                missing_steps = [
                    step
                    for step in expected_steps
                    if step not in detected_steps
                ]

                found_steps = (
                    len(expected_steps)
                    - len(missing_steps)
                )

                compliance_score = (
                    found_steps / len(expected_steps)
                ) * 100

                df = pd.DataFrame(workflow_results)

                st.success(
                    "✅ Workflow Analysis Complete!"
                )

                st.subheader(
                    "📊 Workflow Compliance Summary"
                )

                st.metric(
                    "Compliance Score",
                    f"{compliance_score:.1f}%"
                )

                st.write(
                    f"Detected {found_steps} of "
                    f"{len(expected_steps)} expected steps."
                )

                if missing_steps:

                    st.warning(
                        "⚠️ Missing Workflow Steps Detected"
                    )

                    for step in missing_steps:
                        st.write(f"❌ {step}")

                else:

                    st.success(
                        "✅ All workflow steps detected"
                    )

                st.subheader(
                    "📋 Intelligent Workflow Timeline"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )
