# Project Alignment

## Project Goal

Build an accessible sagittal-video gait analysis system that provides clinicians with repeatable, interpretable measurements for tracking changes in an individual's walking over time.

The primary product hypothesis is:

Standardized sagittal video can provide repeatable, interpretable gait measurements that help clinicians track changes in an individual's walking over time. Biomechanically motivated stability features and composite stability proxies will be developed and validated as extensions of that core longitudinal monitoring capability.

The first marketable MVP should prioritize longitudinal gait monitoring, transparency, repeatability, clinician review, and practical video capture over reproducing laboratory biomechanics exactly.

A longer-term scientific goal is to determine whether sagittal-video-derived biomechanical features can provide a useful and defensible gait-stability proxy that tracks changes in stability over time and identifies potentially unfavorable movement patterns.

The MVP will not claim to measure laboratory-grade Margin of Stability, predict falls, or provide a clinical diagnosis.

The current pipeline already provides the foundation for this goal: monocular video processing, pose estimation and QC, candidate gait-event detection and review, reviewed stride segmentation, and an experimental image-plane COM/body-centroid representation. These remain research proxies rather than validated physical measurements.

## Intended User Experience

A clinician should eventually be able to:

Record or upload a standardized sagittal walking video.
Allow the system to automatically estimate pose, gait events, strides, and relevant gait characteristics.
Review and, when necessary, correct detected gait events or segmentation.
Have affected downstream calculations update automatically.
View a concise set of interpretable gait measurements and quality indicators.
Compare the current assessment with the patient's own prior assessments.
Review changes in timing, variability, asymmetry, body-motion features, and—when sufficiently developed and validated—stability-related metrics.
Share an understandable longitudinal summary with the patient.

The application should expose sufficient QC and visual evidence so that poor capture, tracking problems, or uncertain calculations are visible rather than silently converted into confident results.

The initial product should provide useful longitudinal gait information even if an advanced stability proxy is not yet available for every assessment.

## Primary Use Cases

### Longitudinal gait monitoring

Track whether an individual's measurable gait characteristics remain stable, improve, or deteriorate across repeated assessments.

This may include changes in gait timing, asymmetry, stride-to-stride variability, body-motion patterns, and eventually validated stability-related features.

The primary value is change detection within an individual, rather than requiring a single absolute gait-stability score or claiming that video-derived measurements reproduce laboratory biomechanics.

### Rehabilitation and recovery tracking

Provide clinicians, therapists, researchers, or patients with an additional quantitative signal for observing changes during rehabilitation, recovery, training, or progression of a mobility impairment.

The system should complement—not replace—clinical assessment.

### Identification of potentially unfavorable movement patterns

Highlight changes or patterns that may warrant closer review, such as increasing stride-to-stride variability, asymmetry, altered body-motion behavior, or other stability-related features identified during development and validation.

These should be presented as **signals for review**, not diagnoses or fall predictions.

### Accessible gait research and remote assessment

Provide a lower-cost, easier-to-deploy alternative for exploratory gait monitoring where laboratory motion-capture or force-plate systems are unavailable or impractical.

The product's value comes partly from requiring comparatively simple video acquisition while preserving explicit QC and provenance.

## Value Proposition

Traditional biomechanical gait analysis can require specialized equipment, laboratory space, calibration, trained operators, and significant processing.

This project aims to provide a system that is:

* **Accessible:** based primarily on ordinary RGB video rather than specialized laboratory hardware.
* **Repeatable:** designed for consistent longitudinal measurement under controlled capture conditions.
* **Interpretable:** exposes gait events, trajectories, QC, and contributing features rather than only returning a black-box score.
* **Reviewable:** allows questionable gait events or segmentation to be inspected and eventually corrected interactively.
* **Responsive:** designed to detect meaningful within-person changes over time.
* **Scientifically cautious:** distinguishes video-derived proxies from validated biomechanical measurements.

## What the MVP Is Not

The MVP should not claim to:

* measure laboratory-grade whole-body COM;
* reproduce laboratory-grade Margin of Stability;
* predict falls;
* diagnose neurological, musculoskeletal, or other medical conditions;
* provide a clinical risk classification;
* replace motion capture, force plates, or professional gait assessment.

The current system has no physical scale, calibrated laboratory coordinate system, depth reconstruction, gravity/ground alignment, or reference-system validation, and its gait events and COM proxy remain unvalidated research quantities.

## Scientific Direction

The commercial MVP and the scientific research track should progress together but should not be unnecessarily dependent on each other.

The commercial MVP should focus first on gait measurements that are sufficiently observable, repeatable, interpretable, and useful for longitudinal monitoring.

The scientific development track should investigate whether additional biomechanically motivated features provide meaningful information about gait stability.

Candidate features may include:

gait-event and stride timing;
stride-to-stride variability;
bilateral asymmetry;
body or COM-proxy motion;
body-motion velocity;
foot and support-boundary relationships;
stride-to-stride trajectory consistency;
normalized or image-plane analogues of established stability concepts.

The project should use established biomechanics to motivate these features, but it should not assume that every laboratory quantity must be reconstructed directly from monocular video.

Whole-body COM reconstruction, Margin of Stability, or any particular laboratory-derived quantity should not become a prerequisite for the marketable MVP unless evidence shows that it is necessary for useful longitudinal gait monitoring.

Margin of Stability and related laboratory measures can serve as reference concepts and eventual validation targets, rather than assumptions about what the MVP already measures.

## Validation Philosophy

Validation should progress in stages:

1. **Technical reproducibility** — identical inputs produce deterministic results.
2. **Capture robustness** — reasonable changes in video capture do not cause unexplained output failures.
3. **Test-retest reliability** — similar walking assessments produce similar proxy values.
4. **Sensitivity to meaningful change** — altered gait conditions produce expected changes in relevant features.
5. **Concurrent biomechanical validation** — compare video-derived features against appropriate reference measurements.
6. **Clinical validation**, if pursued later — determine whether changes are meaningful in specific populations or applications.

Scientific validation and product development should proceed together, but lack of final clinical validation should not prevent development of explicitly exploratory candidate metrics.

## Alignment Test for Future Work

Before adding a feature or imposing a new requirement, ask:

Does this work help us deliver useful, repeatable, interpretable longitudinal gait information to clinicians and patients—and, where relevant, advance a defensible stability-related measurement?

For work on the marketable MVP, prioritize capabilities that improve:

reliable sagittal-video analysis;
clinician review and correction;
interpretable gait measurements;
quality control;
comparison with prior assessments;
communication of meaningful changes over time.

For scientific work, prioritize features that improve the robustness, repeatability, responsiveness, or eventual validation of stability-related measurements.

If a proposed requirement substantially increases complexity but is only necessary to reproduce laboratory-grade biomechanics, determine whether it is actually required for the commercial MVP or only for a later scientific objective before allowing it to block progress.

Conversely, speed to market should never justify hiding known uncertainty or making stronger scientific claims than the evidence supports.

The project succeeds first when it delivers useful longitudinal gait monitoring. A validated video-derived stability signal should become an important scientific and product differentiator as the supporting evidence develops.