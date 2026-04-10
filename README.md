# IMBK_Bank_Customer_Churn_ML
---
# 개요
## 1. 프로젝트명
### 고객 이탈 분류 ML 및 인사이트 분석

## 2. 기간
### 2026년 4월 10일

## 3. 기술스택
* **Language:** Python3.10
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, SHAP
* **Machine Learning:** Scikit-learn, PyCaret, LightGBM, Optuna
* **Statistics:** Statsmodels (VIF)

---

# EDA
## 1. 데이터
* **Source:** 캐글 Bank Customer Churn Dataset
* **Shape:** 10,000 Rows × 12 Columns
* **Target:** `churn` (1: 이탈, 0: 유지)

### 📊 데이터 탐색
| 컬럼명 | Dtype | 설명 |
| :--- | :--- | :--- |
| **customer_id** | `int64` | 고객 고유 식별 번호 |
| **credit_score** | `int64` | 고객의 신용 점수 |
| **country** | `object` | 거주 국가 (France, Spain, Germany 등) |
| **gender** | `object` | 성별 (Male / Female) |
| **age** | `int64` | 고객의 연령 |
| **tenure** | `int64` | 은행 거래 기간 (연수) |
| **balance** | `float64` | 계좌 잔고 |
| **products_number** | `int64` | 이용 중인 은행 서비스/상품 개수 |
| **credit_card** | `int64` | 신용카드 보유 여부 (1: 보유, 0: 미보유) |
| **active_member** | `int64` | 활성 회원 여부 (1: 활성, 0: 비활성) |
| **estimated_salary** | `float64` | 추정 연봉 |
| **churn** | `int64` | **(Target)** 이탈 여부 (1: 이탈, 0: 유지) |

### 🔍 데이터 특징 요약
* **결측치 없음:** 모든 컬럼의 Non-Null Count가 10,000개로 데이터 정제 상태 양호.
* **인코딩 필요 변수:** `country`, `gender` (Object 타입) - 모델링 전 전처리 필요.
* **타겟 변수 불균형 확인 필요:** 이탈(`churn`) 여부의 클래스 분포 확인 후 필요시 샘플링 전략 수립 권장.

## 2. 데이터 전처리
* **불필요한 컬럼 제거:** 이탈 예측에 무관한 `customer_id` 제거.
* **범주형 인코딩:** `country`(국가), `gender`(성별) 컬럼 Label Encoding 적용.
    * *Country:* 0 (France), 1 (Germany), 2 (Spain)
    * *Gender:* 0 (Female), 1 (Male)


## 3. EDA 및 해석

### 3.1. 데이터 분포 탐색
<img width="1183" height="1184" alt="hist" src="https://github.com/user-attachments/assets/17e3bd64-e292-45da-af77-fb80a6eb32f7" />

* **종속변수 분포:** `churn` 데이터는 대략 8:2 비율로 클래스 불균형 존재.

### 3.2. Age_Churn 관계도
<img width="1184" height="484" alt="age_churn(Country)" src="https://github.com/user-attachments/assets/d227bf3f-587b-4c4d-b135-045ae7f528e6" />

* **Age와 Churn의 관계 (국가별 꺾은선 그래프):**
    * 20대에서 50대 중반까지는 나이가 들수록 이탈 확률이 증가하는 경향을 보이나, 50대 후반부터는 다시 감소하는 패턴 관찰.
    * 국가별 비교 시 **독일(Germany)**의 고객 이탈율이 전반적으로 가장 높게 나타남. 프랑스와 스페인은 유사한 수준 유지.
    * 프랑스 지역 80~90대에서 평균값이 1로 치솟는 이상값(Outlier) 성향의 패턴 확인.

---

# ML 모델링
## 1. AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value
### 1.1. 변수선정 및 Train_valid_split 
* **다중공선성(Multicollinearity) 해결:** VIF 점수 확인 결과 10 이상으로 나타난 `credit_score` 컬럼 제거하여 모델의 안정성 확보.
* **데이터 분할:** 클래스 불균형(8:2)을 고려하여 `stratify=y` 옵션을 적용한 Train/Validation(8:2) 분할.

### 1.2. AutoML 및 최적화 (Optuna)
* **PyCaret 초기 탐색:** 모든 분류 모델 비교 후 F1 Score 기준 상위 4개 모델 선정 (AdaBoost, LightGBM, GradientBoosting, RandomForest).
* **Optuna 하이퍼파라미터 튜닝:** 각 상위 4개 모델에 대해 Optuna를 활용하여 튜닝 진행.
    * **튜닝 후 성능 (F1 Score):**
        * AdaBoost: 0.5811
        * LightGBM: 0.5939
        * **Gradient Boosting (GBC): 0.6081 (Best Model)**
        * Random Forest: 0.5797

### 1.3. Stacking 파이프라인
* **구조:** Base Models (RF, GBC, LightGBM) + Meta Model (AdaBoost)
* **결과:** Stacking F1 Score **0.5967**. 단일 GBC 모델 성능이 더 우수함을 확인하여 최종적으로 적절한 단일 모델 선정의 중요성을 재확인함.

### 1.4. SHAP Value 사후 분석 (GBC 모델 기준)
<img width="761" height="511" alt="SHAP_value" src="https://github.com/user-attachments/assets/8a1f8694-6a61-475a-97eb-6bb9fe2312ae" />

* **주요 영향 변수 해석:**
    * **이용 상품 수 (`products_number`):** 상품 수가 많을수록 이탈 확률이 대체로 낮아지나, 일부 소수 데이터에서는 높은 이탈 확률을 보임 (고령 고객의 자연 감소 등 외부 요인 추측).
    * **나이 (`age`):** 연령이 높아질수록 이탈에 양(+)의 영향을 미치는 경향이 강함.
    * **활성 상태 (`active_member`):** 비활성 고객일수록 이탈에 강한 양(+)의 영향을 미침.
    * **국가 (`country`):** 1로 인코딩된 독일(Germany) 거주가 이탈에 양(+)의 영향을 줌.

---

# 인사이트 제안
1. **핵심 타겟 마케팅:** EDA 및 SHAP 분석 결과, 연령이 높으면서 이용 중인 상품 수가 적은 고객의 이탈 위험이 큽니다. **50~60대의 적은 상품 이용 고객을 집중 관리 타겟으로 선정**해야 합니다.
2. **활성도 증대 프로모션:** 비활성 상태가 이탈로 직결되므로, 타겟 고객군을 대상으로 신규 상품 가입 시 **우대 금리 적용 등 활성 고객으로 유도하는 공격적인 프로모션**이 필요합니다.
3. **지역 맞춤 전략:** 국가별 이탈률 편차를 고려하여, 신규 유지(Retention) 프로그램의 **시범 운영 국가로 이탈률이 가장 높은 '독일'을 최우선으로 선정**하는 것을 제안합니다.

---

# Reference
* Kaggle Bank Customer Churn Dataset: [Dataset Link] (링크가 있다면 추가해주세요)
* SHAP (SHapley Additive exPlanations) Documentation
* PyCaret Documentation
* Optuna Documentation
