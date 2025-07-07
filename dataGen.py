class SyntheticOpioidDataGenerator:

    def __init__(self, n_patients=1000, max_episode_length=52):
        self.n_patients = n_patients
        self.max_episode_length = max_episode_length  

    def generate_patient_features(self) -> Dict:
        patients = []

        for i in range(self.n_patients):

            age = np.random.normal(55, 15)
            age = np.clip(age, 18, 85)

            initial_dose = np.random.lognormal(3.5, 0.8)  
            initial_dose = np.clip(initial_dose, 10, 300)

            duration_of_use = np.random.exponential(24)  
            duration_of_use = np.clip(duration_of_use, 1, 120)

            depression = np.random.binomial(1, 0.3)
            anxiety = np.random.binomial(1, 0.25)
            substance_abuse_history = np.random.binomial(1, 0.15)
            chronic_pain = np.random.binomial(1, 0.8)

            baseline_pain = np.random.normal(7, 1.5)
            baseline_pain = np.clip(baseline_pain, 1, 10)

            withdrawal_risk = (
                0.3 * (initial_dose / 100) + 
                0.2 * (duration_of_use / 60) + 
                0.2 * depression + 
                0.15 * anxiety + 
                0.15 * substance_abuse_history
            )

            patient = {
                'patient_id': i,
                'age': age,
                'initial_dose': initial_dose,
                'duration_of_use': duration_of_use,
                'depression': depression,
                'anxiety': anxiety,
                'substance_abuse_history': substance_abuse_history,
                'chronic_pain': chronic_pain,
                'baseline_pain': baseline_pain,
                'withdrawal_risk': withdrawal_risk
            }
            patients.append(patient)

        return pd.DataFrame(patients)

    def simulate_episode(self, patient_row: pd.Series) -> List[Dict]:
        episode = []

        current_dose = patient_row['initial_dose']
        current_pain = patient_row['baseline_pain']
        withdrawal_severity = 0
        week = 0

        physician_aggressiveness = np.random.normal(0.15, 0.05)  
        physician_aggressiveness = np.clip(physician_aggressiveness, 0.05, 0.3)

        while week < self.max_episode_length and current_dose > 0:

            state = {
                'week': week,
                'current_dose': current_dose,
                'current_pain': current_pain,
                'withdrawal_severity': withdrawal_severity,
                'age': patient_row['age'],
                'depression': patient_row['depression'],
                'anxiety': patient_row['anxiety'],
                'substance_abuse_history': patient_row['substance_abuse_history'],
                'chronic_pain': patient_row['chronic_pain'],
                'baseline_pain': patient_row['baseline_pain'],
                'withdrawal_risk': patient_row['withdrawal_risk'],
                'dose_change_last_week': 0 if week == 0 else episode[-1]['dose_change']
            }

            if week == 0:
                dose_change = 0  
            else:

                base_reduction = physician_aggressiveness * current_dose

                if current_pain > 8:
                    reduction_factor = 0.5  
                elif withdrawal_severity > 6:
                    reduction_factor = 0.3  
                else:
                    reduction_factor = 1.0

                dose_change = -base_reduction * reduction_factor

                dose_change += np.random.normal(0, base_reduction * 0.1)

                dose_change = max(dose_change, -current_dose)

            new_dose = current_dose + dose_change
            new_dose = max(0, new_dose)

            dose_reduction_ratio = abs(dose_change) / max(current_dose, 1)
            pain_increase = dose_reduction_ratio * np.random.normal(2, 0.5)
            pain_increase = max(0, pain_increase)

            withdrawal_increase = dose_reduction_ratio * np.random.normal(3, 1) * patient_row['withdrawal_risk']
            withdrawal_increase = max(0, withdrawal_increase)

            pain_recovery = np.random.normal(0.1, 0.05)
            withdrawal_recovery = np.random.normal(0.2, 0.1)

            current_pain += pain_increase - pain_recovery
            current_pain = np.clip(current_pain, 0, 10)

            withdrawal_severity += withdrawal_increase - withdrawal_recovery
            withdrawal_severity = np.clip(withdrawal_severity, 0, 10)

            dose_reduction_reward = abs(dose_change) / patient_row['initial_dose'] * 10
            pain_penalty = -(current_pain - patient_row['baseline_pain']) * 2
            withdrawal_penalty = -withdrawal_severity * 1.5

            completion_bonus = 50 if new_dose == 0 else 0

            reward = dose_reduction_reward + pain_penalty + withdrawal_penalty + completion_bonus

            dropout_prob = 0.01 + 0.02 * (current_pain / 10) + 0.03 * (withdrawal_severity / 10)
            terminated = np.random.random() < dropout_prob

            episode_step = {
                **state,
                'dose_change': dose_change,
                'new_dose': new_dose,
                'new_pain': current_pain,
                'new_withdrawal': withdrawal_severity,
                'reward': reward,
                'terminated': terminated,
                'patient_id': patient_row['patient_id']
            }

            episode.append(episode_step)

            current_dose = new_dose
            week += 1

            if terminated or current_dose == 0:
                break

        return episode

    def generate_dataset(self) -> pd.DataFrame:
        patients = self.generate_patient_features()
        all_episodes = []

        print(f"Generating episodes for {self.n_patients} patients...")

        for idx, patient in patients.iterrows():
            episode = self.simulate_episode(patient)
            all_episodes.extend(episode)

            if (idx + 1) % 100 == 0:
                print(f"Generated {idx + 1} patient episodes")

        dataset = pd.DataFrame(all_episodes)
        print(f"Generated {len(dataset)} total transitions")

        return dataset, patients