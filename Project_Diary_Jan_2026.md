# PROJECT DIARY
## Deep Learning-Based Medical Image Analysis for Early Detection of Diabetic Retinopathy

**Project Code**: BITE498J  
**Institution**: VIT University  
**Program**: B.Tech Information Technology  
**Review 1 Date**: 29th January 2026  
**Diary Period**: 1st January 2026 - 28th January 2026

---

## Week 1: January 1-7, 2026

### January 1, 2026 (Wednesday)
**Activity**: Project Initialization and Literature Review  
**Time Spent**: 3 hours  
**Work Done**:
- Reviewed project requirements and assessment criteria for BITE498J
- Identified Diabetic Retinopathy as the project domain
- Began literature survey on deep learning approaches for medical image analysis
- Studied recent papers on DR detection systems (2019-2024)

**Outcomes**:
- Project domain finalized: Medical Image Analysis
- Problem statement drafted
- Initial literature survey started

---

### January 2, 2026 (Thursday)
**Activity**: Dataset Research and Literature Survey  
**Time Spent**: 4 hours  
**Work Done**:
- Researched available diabetic retinopathy datasets
- Identified APTOS 2019 Blindness Detection Dataset (3,662 images)
- Analyzed Kaggle's Diabetic Retinopathy Detection dataset
- Reviewed papers on EfficientNet architecture and transfer learning
- Documented 5 research papers with merits and demerits

**Outcomes**:
- Dataset selected: APTOS 2019
- Understanding of 5-class DR classification problem
- Literature survey: 5 papers documented

**References Reviewed**:
1. Tan & Le (2019) - EfficientNet: Rethinking Model Scaling
2. Gulshan et al. (2016) - Development and Validation of Deep Learning Algorithm
3. Pratt et al. (2016) - Convolutional Neural Networks for DR Detection

---

### January 3, 2026 (Friday)
**Activity**: Architecture Research and Problem Definition  
**Time Spent**: 5 hours  
**Work Done**:
- Studied various CNN architectures (ResNet, DenseNet, EfficientNet)
- Analyzed preprocessing techniques for retinal images
- Researched Ben Graham's preprocessing method
- Studied CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Drafted comprehensive problem statement
- Defined project objectives and scope

**Outcomes**:
- Selected EfficientNet-B4 as base architecture
- Problem definition completed
- Project objectives clearly defined
- Scope document prepared

**Key Decisions**:
- Transfer learning approach selected
- Target metrics defined: Accuracy > 90%, AUC > 0.95

---

### January 4, 2026 (Saturday)
**Activity**: Literature Survey Continuation  
**Time Spent**: 6 hours  
**Work Done**:
- Extended literature survey to 15 papers
- Analyzed papers on:
  - Attention mechanisms in medical imaging
  - Focal Loss for class imbalance
  - Grad-CAM for interpretability
  - Data augmentation strategies
- Created comprehensive comparison table
- Documented merits and demerits of each approach

**Outcomes**:
- Literature survey completed: 15 papers
- Comparative analysis of different approaches
- Identified research gaps
- Understanding of current SOTA methods

**Papers Added**:
4. Lin et al. (2017) - Focal Loss for Dense Object Detection
5. Selvaraju et al. (2017) - Grad-CAM: Visual Explanations
6. Quellec et al. (2017) - Deep Image Mining for DR Screening
7. Wang et al. (2018) - Zoom-in-Net for DR Detection
8-15. Additional papers on CNN architectures and medical imaging

---

### January 5, 2026 (Sunday)
**Activity**: System Architecture Design  
**Time Spent**: 5 hours  
**Work Done**:
- Designed complete system architecture
- Planned preprocessing pipeline:
  - Image resizing to 512×512
  - Ben Graham's preprocessing
  - Circle cropping
  - CLAHE enhancement
- Designed data augmentation strategy
- Planned training methodology (3-phase approach)
- Created architecture flow diagrams

**Outcomes**:
- System architecture designed
- Preprocessing pipeline defined
- Training strategy formulated
- Component integration plan ready

**Architecture Components**:
- Input Layer: RGB fundus images
- Preprocessing: 7-step pipeline
- Augmentation: 6 techniques
- Model: EfficientNet-B4 backbone + custom head
- Output: 5-class predictions + Grad-CAM

---

### January 6, 2026 (Monday)
**Activity**: Project Setup and Environment Configuration  
**Time Spent**: 4 hours  
**Work Done**:
- Created GitHub repository for project
- Set up project directory structure:
  - `/src` - Source code
  - `/notebooks` - Jupyter notebooks
  - `/data` - Dataset storage
  - `/results` - Output and figures
  - `/docs` - Documentation
- Initialized Git version control
- Created README.md with project overview
- Set up requirements.txt with dependencies

**Outcomes**:
- Repository structure established
- Version control initialized
- Project documentation started
- Development environment planned

**Directory Structure Created**:
```
Project-2/
├── src/
├── notebooks/
├── data/raw/
├── data/processed/
├── results/figures/
├── docs/
└── scripts/
```

---

### January 7, 2026 (Tuesday)
**Activity**: Requirements Documentation and Methodology Planning  
**Time Spent**: 4 hours  
**Work Done**:
- Documented technical requirements
- Created requirements.txt with all Python dependencies:
  - PyTorch 2.0
  - torchvision
  - OpenCV
  - Albumentations
  - scikit-learn
  - matplotlib, seaborn
- Planned methodology for Review 1
- Created project proposal document
- Drafted abstract (200+ words)

**Outcomes**:
- Technical requirements documented
- Dependencies listed
- Methodology section drafted
- Project proposal prepared

---

## Week 2: January 8-14, 2026

### January 8, 2026 (Wednesday)
**Activity**: Dataset Acquisition and Initial Analysis  
**Time Spent**: 5 hours  
**Work Done**:
- Downloaded APTOS 2019 dataset from Kaggle
- Analyzed dataset structure and class distribution
- Created dataset documentation
- Identified class imbalance issue:
  - Class 0 (No DR): 1,805 images (49.3%)
  - Class 1 (Mild): 370 images (10.1%)
  - Class 2 (Moderate): 999 images (27.3%)
  - Class 3 (Severe): 193 images (5.3%)
  - Class 4 (Proliferative): 295 images (8.1%)
- Created download script: `download_dataset.py`

**Outcomes**:
- Dataset acquired (3,662 training images)
- Class imbalance documented
- Need for weighted sampling identified
- Dataset statistics computed

**Challenges Identified**:
- Severe class imbalance
- Variable image quality
- Different image resolutions

---

### January 9, 2026 (Thursday)
**Activity**: Exploratory Data Analysis (EDA)  
**Time Spent**: 6 hours  
**Work Done**:
- Created Jupyter notebook: `01_EDA_Preprocessing.ipynb`
- Performed comprehensive EDA:
  - Image resolution analysis
  - Color distribution analysis
  - Class-wise visualization
  - Quality assessment
- Generated EDA visualizations
- Documented findings

**Outcomes**:
- EDA notebook completed
- Image characteristics understood
- Preprocessing needs identified
- Visualization plots generated

**Key Findings**:
- Image resolutions vary: 400px to 5,184px
- Many images have black borders
- Variable lighting conditions
- Need for normalization

---

### January 10, 2026 (Friday)
**Activity**: Preprocessing Pipeline Development  
**Time Spent**: 5 hours  
**Work Done**:
- Implemented preprocessing functions in `preprocessing.py`:
  - Black border cropping
  - Image resizing
  - Ben Graham's method implementation
  - Circle cropping
  - CLAHE enhancement
- Created test preprocessing pipeline
- Validated preprocessing results
- Documented preprocessing steps

**Outcomes**:
- Preprocessing module created
- 7-step pipeline implemented
- Quality improvements verified
- Code documented

**Preprocessing Steps Implemented**:
1. Quality assessment
2. Black border removal
3. Resizing to 512×512
4. Ben Graham's preprocessing
5. Circle cropping
6. CLAHE enhancement
7. Normalization [0, 1]

---

### January 11, 2026 (Saturday)
**Activity**: Data Augmentation Strategy  
**Time Spent**: 4 hours  
**Work Done**:
- Researched augmentation techniques for medical images
- Implemented augmentation pipeline using Albumentations
- Created augmentation functions:
  - Random rotation (±30°)
  - Horizontal/vertical flipping
  - Brightness/contrast adjustment
  - Random zoom (0.9-1.1)
  - Gaussian noise addition
  - CoarseDropout
- Tested augmentation pipeline
- Validated augmented images

**Outcomes**:
- Augmentation module ready
- 6 augmentation techniques implemented
- Medical image compatibility ensured
- Test augmentations successful

---

### January 12, 2026 (Sunday)
**Activity**: Model Architecture Design and Planning  
**Time Spent**: 5 hours  
**Work Done**:
- Studied EfficientNet-B4 architecture in detail
- Designed custom classification head:
  - Dropout(0.4) → Dense(512, ReLU)
  - BatchNorm → Dropout(0.3)
  - Dense(5, Softmax)
- Planned 3-phase training strategy:
  - Phase 1: Feature extraction (10 epochs)
  - Phase 2: Fine-tuning top 50% (20 epochs)
  - Phase 3: Full fine-tuning (10 epochs)
- Researched Focal Loss for class imbalance
- Planned evaluation metrics

**Outcomes**:
- Model architecture finalized
- Classification head designed
- Training strategy planned
- Loss function selected: Focal Loss

**Model Specifications**:
- Backbone: EfficientNet-B4 (pretrained on ImageNet)
- Feature dimensions: 1792
- Custom head: 2 fully connected layers
- Total trainable parameters: ~19M

---

### January 13, 2026 (Monday)
**Activity**: Configuration and Dataset Module  
**Time Spent**: 4 hours  
**Work Done**:
- Created `config.py` with all hyperparameters:
  - Learning rates: 1e-3, 1e-4, 1e-5
  - Batch size: 32
  - Image size: 512×512
  - Dropout rates: 0.4, 0.3
  - Focal Loss parameters: γ=2.0, α=balanced
- Implemented `dataset.py`:
  - Custom PyTorch Dataset class
  - Data loading pipeline
  - Train/validation split (80/20)
  - Weighted random sampling
- Tested dataset module
- Validated data loading

**Outcomes**:
- Configuration module created
- Dataset module implemented
- Data loading verified
- Sampling strategy tested

---

### January 14, 2026 (Tuesday)
**Activity**: Grad-CAM Implementation Research  
**Time Spent**: 4 hours  
**Work Done**:
- Studied Grad-CAM (Gradient-weighted Class Activation Mapping)
- Researched interpretability in medical AI
- Planned Grad-CAM integration with EfficientNet-B4
- Created implementation plan for visualization
- Documented interpretability requirements
- Reviewed clinical explainability needs

**Outcomes**:
- Grad-CAM approach understood
- Implementation plan ready
- Interpretability strategy defined
- Clinical validation approach planned

**Interpretability Features Planned**:
- Heatmap overlays on original images
- Class-specific activation maps
- Confidence scores
- Region of interest highlighting

---

## Week 3: January 15-21, 2026

### January 15, 2026 (Wednesday)
**Activity**: Training Pipeline Planning  
**Time Spent**: 5 hours  
**Work Done**:
- Designed complete training pipeline
- Planned learning rate scheduling:
  - Phase 1: Constant LR=1e-3
  - Phase 2: Constant LR=1e-4
  - Phase 3: Constant LR=1e-5
- Researched gradient clipping strategies
- Planned checkpoint saving strategy
- Designed early stopping mechanism
- Created training monitoring plan

**Outcomes**:
- Training pipeline designed
- LR strategy finalized
- Checkpoint system planned
- Monitoring metrics defined

**Training Monitoring**:
- Loss curves (train/val)
- Accuracy curves
- Confusion matrix
- Per-class metrics
- Learning rate tracking

---

### January 16, 2026 (Thursday)
**Activity**: Evaluation Metrics Implementation  
**Time Spent**: 4 hours  
**Work Done**:
- Implemented evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-score (per class)
  - AUC-ROC score
  - Quadratic Weighted Kappa (QWK)
  - Sensitivity and Specificity
  - Confusion matrix
- Created metrics calculation module
- Tested metric functions
- Planned result visualization

**Outcomes**:
- Metrics module completed
- 8 evaluation metrics implemented
- Testing successful
- Visualization plan ready

**Target Metrics**:
- Accuracy: > 90%
- AUC-ROC: > 0.95
- QWK: > 0.85
- Sensitivity: > 85% (per class)

---

### January 17, 2026 (Friday)
**Activity**: Review 1 Documentation Preparation  
**Time Spent**: 6 hours  
**Work Done**:
- Started preparing Review 1 report
- Drafted comprehensive abstract (250 words)
- Wrote detailed introduction section
- Documented problem statement clearly
- Listed all 4 project objectives
- Defined project scope and limitations
- Created proposed system description
- Compiled 15-paper literature survey

**Outcomes**:
- Report structure created
- Abstract completed
- Introduction written
- Problem statement finalized
- Objectives documented

**Report Sections Completed**:
1. Title & Abstract
2. Introduction
3. Problem Statement
4. Objectives (4 main objectives)
5. Scope of the Project

---

### January 18, 2026 (Saturday)
**Activity**: Literature Survey Table and System Architecture  
**Time Spent**: 7 hours  
**Work Done**:
- Created comprehensive literature survey table:
  - 15 research papers
  - Columns: S.NO, TITLE, MERITS, DEMERITS
  - Covered years 2016-2024
  - Included CNN architectures, loss functions, interpretability
- Designed system architecture diagram
- Documented architecture components
- Created data flow description
- Wrote architecture explanation

**Outcomes**:
- Literature survey table completed (15 entries)
- System architecture documented
- Component descriptions written
- Data flow explained

**Literature Survey Coverage**:
- CNN architectures (5 papers)
- Transfer learning (3 papers)
- Loss functions (2 papers)
- Interpretability methods (2 papers)
- Medical imaging specific (3 papers)

---

### January 19, 2026 (Sunday)
**Activity**: Timeline and References  
**Time Spent**: 5 hours  
**Work Done**:
- Created detailed project timeline:
  - Phase 1: Jan-Feb (Data collection, preprocessing)
  - Phase 2: Feb-Mar (Model development, training)
  - Phase 3: Mar-Apr (Evaluation, optimization)
  - Phase 4: Apr (Deployment, documentation)
- Compiled 15 references in APA format
- Organized references by category
- Verified all citations
- Cross-checked bibliography

**Outcomes**:
- Timeline created for 4 months
- 15 references compiled
- APA formatting verified
- Citation consistency checked

**Timeline Milestones**:
- Review 1: 29 Jan 2026
- Review 2: Mid-March 2026
- Review 3: End-April 2026
- Final Submission: May 2026

---

### January 20, 2026 (Monday)
**Activity**: Presentation Planning and Slide Creation  
**Time Spent**: 6 hours  
**Work Done**:
- Planned 20-slide presentation structure
- Created slide outline:
  - Title slide
  - Agenda
  - Introduction to Diabetic Retinopathy
  - Global statistics
  - Problem statement
  - Motivation
  - Literature survey (2 slides)
  - Research gap
  - Objectives
  - Scope
  - Dataset overview
  - System architecture
  - Preprocessing details
  - Model architecture
  - Training strategy
  - Grad-CAM explanation
  - Evaluation metrics
  - Expected outcomes
  - Timeline
  - Conclusion
- Started creating presentation content

**Outcomes**:
- 20-slide outline created
- Content structure finalized
- Slide flow organized
- Visual elements planned

---

### January 21, 2026 (Tuesday)
**Activity**: Presentation Content Development  
**Time Spent**: 7 hours  
**Work Done**:
- Developed content for all 20 slides
- Created visual descriptions for:
  - DR severity progression
  - System architecture diagram
  - Preprocessing pipeline
  - Model architecture
  - Training phases
  - Grad-CAM visualization
- Added key statistics and facts
- Included technical specifications
- Prepared speaker notes
- Reviewed presentation flow

**Outcomes**:
- All 20 slides content completed
- Visual elements described
- Technical details included
- Presentation ready for generation

**Presentation Highlights**:
- Comprehensive DR introduction
- Complete literature review summary
- Detailed technical approach
- Clear methodology explanation
- Realistic timeline

---

## Week 4: January 22-28, 2026

### January 22, 2026 (Wednesday)
**Activity**: Report Finalization and Formatting  
**Time Spent**: 6 hours  
**Work Done**:
- Finalized complete Review 1 report
- Formatted document according to university guidelines:
  - Times New Roman font
  - 12pt body text
  - 15pt section headers
  - 14pt institution name
  - Justified alignment
- Created title page with university logo
- Added table of contents
- Formatted literature survey table
- Added page numbers
- Reviewed all sections

**Outcomes**:
- Complete report ready
- Proper formatting applied
- University guidelines followed
- Professional appearance achieved

**Report Statistics**:
- Total pages: 12-15
- Word count: ~3,500 words
- Sections: 9 major sections
- Tables: 1 (literature survey)
- Diagrams: 1 (system architecture)

---

### January 23, 2026 (Thursday)
**Activity**: PowerPoint Presentation Generation  
**Time Spent**: 5 hours  
**Work Done**:
- Generated PowerPoint presentation using Python (python-pptx)
- Created all 20 slides with proper layout
- Applied professional design template
- Added university branding
- Included all technical diagrams
- Added bullet points and content
- Formatted text appropriately
- Reviewed slide transitions

**Outcomes**:
- 20-slide PowerPoint generated
- Professional design applied
- All content included
- Ready for presentation

**Presentation Format**:
- Title slide with university logo
- Consistent color scheme
- Clear section dividers
- Technical diagrams included
- Proper citations

---

### January 24, 2026 (Friday)
**Activity**: System Architecture Diagram Creation  
**Time Spent**: 5 hours  
**Work Done**:
- Created visual system architecture diagram using matplotlib
- Designed vertical layout showing complete pipeline:
  - Input layer
  - Preprocessing pipeline (7 steps)
  - Data augmentation (6 techniques)
  - Deep learning model (EfficientNet-B4)
  - Classification head
  - Output layer with Grad-CAM
- Added color coding for different components
- Included all technical specifications
- Created high-resolution version (600 DPI)
- Generated both PNG versions

**Outcomes**:
- System architecture diagram created
- High-resolution image generated
- Professional appearance
- All components visualized

**Challenges**:
- Initial diagram was too large (14×18 inches)
- Occupied excessive space
- Needed optimization for report

---

### January 25, 2026 (Saturday)
**Activity**: Architecture Diagram Optimization  
**Time Spent**: 4 hours  
**Work Done**:
- Redesigned architecture diagram for compact layout
- Changed from vertical (14×18) to horizontal (16×6) design
- Reduced space by 70%
- Organized into 3 rows:
  - Row 1: Main pipeline (Input → Preprocessing → Augmentation → Model → Output)
  - Row 2: Technical details (Classification Head, Training, Loss, Metrics)
  - Row 3: Infrastructure (Dataset, Tech Stack, Deployment)
- Optimized font sizes (7-8pt)
- Improved visual clarity
- Generated compact versions (300 DPI, 400 DPI)

**Outcomes**:
- Compact diagram created (16×6 inches)
- 70% space reduction achieved
- Better organization
- Crisp, professional appearance
- Files: system_architecture.png (360KB), system_architecture_hires.png (500KB)

**Improvements**:
- Horizontal flow more intuitive
- Better space utilization
- Easier to read
- More information density
- Print-friendly format

---

### January 26, 2026 (Sunday)
**Activity**: Documentation Review and Code Organization  
**Time Spent**: 5 hours  
**Work Done**:
- Reviewed all project documentation
- Organized code files:
  - `create_report_docx.py` - Report generator
  - `create_presentation.py` - PowerPoint generator
  - `create_architecture_diagram.py` - Diagram generator
  - `download_dataset.py` - Dataset download script
  - `preprocessing.py` - Preprocessing module
  - `dataset.py` - Dataset module
  - `config.py` - Configuration
- Updated README.md
- Verified all file paths
- Tested all generation scripts
- Committed changes to Git

**Outcomes**:
- Documentation reviewed
- Code organized
- Scripts tested
- Repository updated

**Files Organized**:
- Python scripts: 7 files
- Documentation: 5 files
- Notebooks: 1 file
- Generated outputs: 3 files

---

### January 27, 2026 (Monday)
**Activity**: Final Review and Quality Assurance  
**Time Spent**: 6 hours  
**Work Done**:
- Performed final review of all deliverables:
  - Review 1 Report (Word document)
  - PowerPoint Presentation (20 slides)
  - System Architecture Diagram
- Verified formatting consistency
- Checked all technical details
- Validated references (15 papers)
- Reviewed literature survey table
- Verified objectives alignment
- Checked scope definition
- Validated architecture description
- Proofread all content

**Outcomes**:
- Quality assurance completed
- All deliverables verified
- Content accuracy confirmed
- Formatting validated
- Ready for Review 1

**Quality Checks Performed**:
✓ Spelling and grammar
✓ Technical accuracy
✓ Reference formatting (APA)
✓ Visual quality
✓ Content completeness
✓ University guidelines compliance

---

### January 28, 2026 (Tuesday)
**Activity**: Final Preparation and Backup  
**Time Spent**: 4 hours  
**Work Done**:
- Created backup of all project files
- Generated final versions:
  - Review_1_Report.docx (final)
  - First_Review_Presentation_20_Slides.pptx (final)
  - system_architecture.png (compact)
  - system_architecture_hires.png (print quality)
- Prepared presentation notes
- Organized supporting documents
- Reviewed presentation flow
- Practiced presentation timing
- Committed final changes to repository

**Outcomes**:
- All deliverables finalized
- Backups created
- Presentation rehearsed
- Ready for 29th January Review 1

**Final Deliverables**:
1. ✅ Review 1 Report (Word format, 12-15 pages)
2. ✅ PowerPoint Presentation (20 slides)
3. ✅ System Architecture Diagram (compact, high-quality)
4. ✅ Project Diary (this document)

---

## Summary of Work Done (January 1-28, 2026)

### Week-wise Distribution:
- **Week 1 (Jan 1-7)**: Literature review, problem definition, architecture design
- **Week 2 (Jan 8-14)**: Dataset acquisition, EDA, preprocessing, model planning
- **Week 3 (Jan 15-21)**: Training planning, metrics, documentation, presentation
- **Week 4 (Jan 22-28)**: Finalization, generation, optimization, quality assurance

### Total Hours: **132 hours**

### Key Achievements:
✅ Comprehensive literature survey (15 papers)  
✅ Complete system architecture designed  
✅ Preprocessing pipeline implemented  
✅ Training strategy planned  
✅ Review 1 Report completed (12-15 pages)  
✅ PowerPoint presentation created (20 slides)  
✅ System architecture diagram generated  
✅ Dataset acquired and analyzed (3,662 images)  
✅ Project repository organized  
✅ All documentation prepared  

### Technologies/Tools Used:
- **Languages**: Python 3.10+
- **Frameworks**: PyTorch 2.0
- **Libraries**: OpenCV, Albumentations, matplotlib, scikit-learn
- **Documentation**: Microsoft Word, PowerPoint, Markdown
- **Version Control**: Git/GitHub
- **Visualization**: matplotlib, seaborn
- **Development**: Jupyter Notebook, VS Code

### Challenges Faced and Solutions:
1. **Class Imbalance** → Solution: Focal Loss + Weighted Sampling
2. **Variable Image Quality** → Solution: 7-step preprocessing pipeline
3. **Large Diagram Size** → Solution: Redesigned to compact horizontal layout
4. **Literature Survey** → Solution: Systematic review of 15 papers
5. **Architecture Design** → Solution: Transfer learning with EfficientNet-B4

### Skills Developed:
- Medical image analysis
- Deep learning architecture design
- Transfer learning techniques
- Research paper analysis
- Technical documentation
- Academic presentation
- Project management

---

## Readiness for Review 1 (29th January 2026)

### Assessment Criteria Coverage:

**1. Title & Abstract [5 Marks]**
✅ Comprehensive title clearly stating project scope  
✅ 250-word abstract covering problem, approach, expected outcomes  

**2. Problem Definition & Literature Survey [6 Marks]**
✅ Clear problem statement with context and impact  
✅ 15-paper literature survey with comparative analysis  
✅ Merits and demerits documented for each paper  
✅ Research gap identified  

**3. Objectives & Scope [4 Marks]**
✅ 4 clearly defined objectives with measurable outcomes  
✅ Comprehensive scope definition  
✅ Limitations clearly stated  

**4. Proposed Architecture [5 Marks]**
✅ Detailed system architecture diagram  
✅ Complete component descriptions  
✅ Data flow explained  
✅ Technical specifications provided  

**5. Report & Presentation [Mandatory]**
✅ Professional Word document (12-15 pages)  
✅ 20-slide PowerPoint presentation  
✅ Proper formatting and university guidelines  
✅ High-quality visuals and diagrams  

---

## Next Steps (Post Review 1)

### Phase 2: Implementation (February-March 2026)
1. Complete model implementation
2. Train EfficientNet-B4 model (3 phases)
3. Implement Grad-CAM visualization
4. Perform comprehensive evaluation
5. Optimize hyperparameters
6. Document results

### Phase 3: Evaluation (March-April 2026)
1. Test on validation set
2. Generate confusion matrix
3. Calculate all metrics
4. Compare with baseline models
5. Perform error analysis
6. Prepare Review 2 documentation

### Phase 4: Deployment (April 2026)
1. Create web interface (Streamlit)
2. Optimize inference time
3. User testing
4. Documentation
5. Final presentation preparation

---

## Mentor Meetings and Feedback

### Meeting 1 (January 10, 2026)
- Discussed project topic selection
- Approved Diabetic Retinopathy focus
- Suggested EfficientNet architecture
- Recommended APTOS 2019 dataset

### Meeting 2 (January 17, 2026)
- Reviewed literature survey
- Approved system architecture
- Discussed preprocessing techniques
- Confirmed training strategy

### Meeting 3 (January 24, 2026)
- Reviewed draft report
- Approved presentation structure
- Provided formatting guidelines
- Confirmed Review 1 readiness

---

## Personal Reflection

This intensive month of work has provided deep insights into:
- Medical AI applications and their societal impact
- Deep learning model design and training strategies
- Importance of thorough literature review
- Technical documentation and academic writing
- Time management for complex projects
- Problem-solving and iterative refinement

The project has strengthened my understanding of:
- Transfer learning and fine-tuning techniques
- Handling class imbalance in medical datasets
- Interpretability in AI systems (Grad-CAM)
- Professional documentation practices

Looking forward to implementing the designed system and achieving the target performance metrics in the coming months.

---

**Prepared by**: [Register Number]  
**Guide**: [Faculty Name]  
**Date**: 28th January 2026  
**Next Review**: Review 2 (Mid-March 2026)

---

*End of Project Diary - January 2026*
