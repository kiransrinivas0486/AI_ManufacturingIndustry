from openai import OpenAI
import streamlit as st
import cv2
import base64
import pandas as pd
import tempfile
import time

# -----------------------------------
# OpenAI Client ̰
# -----------------------------------
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)
# -----------------------------------
# Streamlit Config
# -----------------------------------
st.set_page_config(
    page_title="Manufacturing Workflow AI",
    layout="wide"
)

st.title("🏭 Manufacturing Workflow Intelligence")

st.write(
    "Upload a machining workflow video for AI-powered process analysis."
)

# -----------------------------------
# Upload Video
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Manufacturing Video",
    type=["mp4", "mov", "avi"]
)

# -----------------------------------
# If File Uploaded
# -----------------------------------
if uploaded_file is not None:

    st.subheader("📹 Uploaded Video")

    st.video(uploaded_file)

    # -----------------------------------
    # Analyze Button
    # -----------------------------------
    if st.button("Analyze Workflow"):

        with st.spinner("Analyzing workflow..."):

            # -----------------------------------
            # Save Video Temporarily
            # -----------------------------------
            temp_video = tempfile.NamedTemporaryFile(
                delete=False
            )

            temp_video.write(
                uploaded_file.read()
            )

            video_path = temp_video.name

            # -----------------------------------
            # Open Video
            # -----------------------------------
            cap = cv2.VideoCapture(video_path)

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            st.write(f"🎥 FPS Detected: {fps}")

            # -----------------------------------
            # Sample Every 3 Seconds
            # -----------------------------------
            frame_interval = int(fps * 3)

            frame_count = 0

            # -----------------------------------
            # Safety Limit
            # -----------------------------------
            max_frames = 25

            processed_frames = 0

            # -----------------------------------
            # Temporary Raw Step Storage
            # -----------------------------------
            raw_steps = []

            # -----------------------------------
            # Read Video Frames
            # -----------------------------------
            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # -----------------------------------
                # Process Frame Every 3 Seconds
                # -----------------------------------
                if frame_count % frame_interval == 0:

                    timestamp = (
                        frame_count / fps
                    )

                    st.write(
                        f"Analyzing frame at "
                        f"{int(timestamp)} sec..."
                    )

                    # -----------------------------------
                    # Resize Smaller
                    # -----------------------------------
                    frame = cv2.resize(
                        frame,
                        (640, 360)
                    )

                    # -----------------------------------
                    # Save Frame
                    # -----------------------------------
                    frame_filename = (
                        f"frame_{int(timestamp)}.jpg"
                    )

                    cv2.imwrite(
                        frame_filename,
                        frame
                    )

                    # -----------------------------------
                    # Convert To Base64
                    # -----------------------------------
                    with open(
                        frame_filename,
                        "rb"
                    ) as image_file:

                        image_base64 = (
                            base64.b64encode(
                                image_file.read()
                            ).decode("utf-8")
                        )

                    # -----------------------------------
                    # Send To GPT Vision
                    # -----------------------------------
                    response = (
                        client.chat.completions.create(

                            model="gpt-4o-mini",

                            messages=[

                                {
                                    "role": "system",
                                    "content": """
You are an industrial machining workflow expert.

The video shows a worker performing this workflow:

1. Pick metal component
2. Load component into machine
3. Machining process
4. Remove machined component
5. Apply green paint/coating
6. Place component into tray

Your task:
- Identify EXACTLY what step is happening
- Be VERY specific
- Detect green paint application carefully
- Detect tray placement carefully
- Avoid generic manufacturing language

IMPORTANT:
Only choose ONE of these step names:

- Component Pickup
- Machine Loading
- Machining Process
- Part Removal
- Green Paint Application
- Tray Placement

Return STRICTLY:

Step Name: <step>

Detailed Analysis: <specific action>
"""
                                },

                                {
                                    "role": "user",
                                    "content": [

                                        {
                                            "type": "text",
                                            "text": (
                                                "Analyze this "
                                                "workflow frame."
                                            )
                                        },

                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url":
                                                (
                                                    f"data:image/jpeg;base64,"
                                                    f"{image_base64}"
                                                )
                                            }
                                        }
                                    ]
                                }
                            ]
                        )
                    )

                    # -----------------------------------
                    # Get AI Response
                    # -----------------------------------
                    analysis = (
                        response.choices[0]
                        .message.content
                    )

                    # -----------------------------------
                    # Prevent Rate Limit
                    # -----------------------------------
                    time.sleep(2)

                    # -----------------------------------
                    # Parse AI Response
                    # -----------------------------------
                    step_name = "Unknown"

                    detailed_analysis = analysis

                    try:

                        lines = analysis.split("\n")

                        for line in lines:

                            if line.startswith(
                                "Step Name:"
                            ):

                                step_name = (
                                    line.replace(
                                        "Step Name:",
                                        ""
                                    ).strip()
                                )

                            elif line.startswith(
                                "Detailed Analysis:"
                            ):

                                detailed_analysis = (
                                    line.replace(
                                        "Detailed Analysis:",
                                        ""
                                    ).strip()
                                )

                    except:
                        pass

                    # -----------------------------------
                    # Store RAW Steps
                    # -----------------------------------
                    raw_steps.append({

                        "timestamp":
                            timestamp,

                        "step_name":
                            step_name,

                        "analysis":
                            detailed_analysis
                    })

                    # -----------------------------------
                    # Processed Count
                    # -----------------------------------
                    processed_frames += 1

                    # -----------------------------------
                    # Stop After Max Frames
                    # -----------------------------------
                    if processed_frames >= max_frames:
                        break

                frame_count += 1

            # -----------------------------------
            # Release Video
            # -----------------------------------
            cap.release()

            # -----------------------------------
            # Build Workflow Stages
            # -----------------------------------
            workflow_results = []

            if len(raw_steps) > 0:

                current_step = raw_steps[0]

                start_time = (
                    current_step["timestamp"]
                )

                current_name = (
                    current_step["step_name"]
                )

                current_analysis = (
                    current_step["analysis"]
                )

                for i in range(1, len(raw_steps)):

                    next_step = raw_steps[i]

                    # -----------------------------------
                    # If Step Changes
                    # -----------------------------------
                    if (
                        next_step["step_name"]
                        != current_name
                    ):

                        end_time = (
                            next_step["timestamp"]
                        )

                        duration = (
                            end_time - start_time
                        )

                        workflow_results.append({

                            "Start Time":
                                f"{int(start_time)} sec",

                            "End Time":
                                f"{int(end_time)} sec",

                            "Duration":
                                f"{int(duration)} sec",

                            "Step Name":
                                current_name,

                            "Detailed Analysis":
                                current_analysis
                        })

                        # Start New Step
                        current_step = next_step

                        start_time = (
                            next_step["timestamp"]
                        )

                        current_name = (
                            next_step["step_name"]
                        )

                        current_analysis = (
                            next_step["analysis"]
                        )

                # -----------------------------------
                # Add Final Step
                # -----------------------------------
                workflow_results.append({

                    "Start Time":
                        f"{int(start_time)} sec",

                    "End Time":
                        "End",

                    "Duration":
                        "Ongoing",

                    "Step Name":
                        current_name,

                    "Detailed Analysis":
                        current_analysis
                })
                            # -----------------------------------
            # Expected Workflow
            # -----------------------------------
            expected_steps = [
                "Component Pickup",
                "Machine Loading",
                "Machining Process",
                "Part Removal",
                "Green Paint Application",
                "Tray Placement"
            ]

            # -----------------------------------
            # Extract Detected Steps
            # -----------------------------------
            detected_steps = []

            for step in workflow_results:
                detected_steps.append(
                    step["Step Name"]
                )

            # -----------------------------------
            # Find Missing Steps
            # -----------------------------------
            missing_steps = []

            for step in expected_steps:

                if step not in detected_steps:
                    missing_steps.append(step)

            # -----------------------------------
            # Compliance Score
            # -----------------------------------
            found_steps = (
                len(expected_steps)
                - len(missing_steps)
            )

            compliance_score = (
                found_steps
                / len(expected_steps)
            ) * 100

            # -----------------------------------
            # Display Results
            # -----------------------------------
            df = pd.DataFrame(
                workflow_results
            )

            st.success(
                "✅ Workflow Analysis Complete!"
            )

            # -----------------------------------
            # Workflow Summary
            # -----------------------------------
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

            # -----------------------------------
            # Missing Steps
            # -----------------------------------
            if missing_steps:

                st.warning(
                    "⚠️ Missing Workflow Steps Detected"
                )

                for step in missing_steps:

                    st.write(
                        f"❌ {step}"
                    )

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