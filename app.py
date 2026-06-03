import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

st.title("🏢 Building Change Detection System")
st.write("Upload before and after satellite images to detect building changes.")

# Timestamp
st.write(f"🕒 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Upload images
before_file = st.file_uploader("Upload BEFORE image", type=["png", "jpg", "jpeg"])
after_file = st.file_uploader("Upload AFTER image", type=["png", "jpg", "jpeg"])

if before_file and after_file:

    # Load images
    before = np.array(Image.open(before_file).convert("RGB"))
    after = np.array(Image.open(after_file).convert("RGB"))

    st.subheader("Input Images")
    col1, col2 = st.columns(2)
    col1.image(before, caption="Before Image")
    col2.image(after, caption="After Image")

    # Resize
    after = cv2.resize(after, (before.shape[1], before.shape[0]))

    # ----------------------------------------
    # CHANGE DETECTION
    # ----------------------------------------
    diff = cv2.absdiff(before, after)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    final_mask = np.zeros_like(clean)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1500:
            final_mask[labels == i] = 255

    # ----------------------------------------
    # AREA CALCULATION
    # ----------------------------------------
    changed_pixels = np.sum(final_mask > 0)

    meter_per_pixel = 0.5
    area_m2 = changed_pixels * (meter_per_pixel ** 2)
    area_sqft = area_m2 * 10.764

    # ----------------------------------------
    # OVERLAY
    # ----------------------------------------
    overlay = after.copy()
    overlay[final_mask > 0] = [255, 0, 0]

    # ----------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------
    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.image(final_mask, caption="Detected Change Mask")
    col2.image(overlay, caption="Change Overlay")

    # ----------------------------------------
    # ANALYSIS
    # ----------------------------------------
    st.subheader("📊 Change Analysis")
    st.write(f"Changed Pixels: {changed_pixels}")
    st.write(f"Area: {area_m2:.2f} sq. meters")
    st.write(f"Area: {area_sqft:.2f} sq. feet")

    # Severity level
    severity = "Low"
    if changed_pixels > 200000:
        severity = "High"
    elif changed_pixels > 50000:
        severity = "Medium"

    st.write(f"**Severity Level:** {severity}")

    # Download report
    report_text = f"""
Building Change Detection Report

Changed Pixels: {changed_pixels}
Area (sq. meters): {area_m2:.2f}
Area (sq. feet): {area_sqft:.2f}
Severity Level: {severity}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    st.download_button(
        "📄 Download Change Report",
        data=report_text,
        file_name="change_report.txt",
        mime="text/plain"
    )

    # ----------------------------------------
    # 🚨 AUTHORITY ALERT
    # ----------------------------------------
    st.subheader("🚨 Authority Alert")

    if changed_pixels > 200000:
        st.error("HIGH ALERT: Major building-related change detected.")
        st.write("**Recommended Action:** Immediate authority verification required.")
        st.write("**Next Step:** Cross-check with municipal approval / land records.")

    elif changed_pixels > 50000:
        st.warning("MEDIUM ALERT: Noticeable building-related change detected.")
        st.write("**Recommended Action:** Field inspection recommended.")
        st.write("**Next Step:** Verify whether construction or demolition is authorized.")

    else:
        st.info("LOW ALERT: Minor building-related change detected.")
        st.write("**Recommended Action:** Monitor the area for further changes.")

    # ----------------------------------------
    # 📩 OWNER NOTICE (PROPOSED)
    # ----------------------------------------
    st.subheader("📩 Owner Notice (Proposed)")

    verification_status = st.selectbox(
        "Authority Verification Status",
        ["Pending Verification", "Authorized Construction", "Unauthorized Construction"]
    )

    owner_name = st.text_input("Registered Owner Name", value="Mr./Ms. Owner")
    property_id = st.text_input("Property ID", value="PROP-102")
    location_name = st.text_input("Location / Zone", value="Secunderabad, Telangana")

    if verification_status == "Pending Verification":
        st.info("Owner notice is not generated until authority verification is completed.")

    elif verification_status == "Authorized Construction":
        st.success("Construction verified as authorized. No notice required.")

    elif verification_status == "Unauthorized Construction":
        st.error("Unauthorized construction suspected after authority verification.")

        notice_text = f"""
Notice to Registered Owner: {owner_name}

Property ID: {property_id}
Location: {location_name}

Building-related change has been detected in the monitored satellite imagery.
After authority verification, this construction is marked as potentially unauthorized.

You are requested to submit valid approval / permit documents to the municipal authority.
Failure to respond may lead to further legal action.

- Municipal Monitoring Authority
"""

        st.text_area("Generated Notice", notice_text, height=220)

        st.download_button(
            "📥 Download Notice",
            data=notice_text,
            file_name="owner_notice.txt",
            mime="text/plain"
        )

    # ----------------------------------------
    # NOTE
    # ----------------------------------------
    st.caption("Note: Legal verification requires integration with government land and permit records.")
