Full analysis scripts for the 5-week characterization cohorts and the weight data for the running and stopping task cohorts.
This list contains documentation for the types of data and analyses each script works with. There are a lot of old analysis functions in each script that were part of the original iterative process to find better data models. We have left them in the scripts although many are not directly used in analyses for the paper. The beginning of each script contains details on the packages used and tests available.

1. across_cohort.py and behavioral_analysis.py both handle the daily and total change and homecage behavioral metrics data types for the characterization cohorts. Any reference to ramp unspecified is typically to the SR, not the FR.
2. CAH and RV weight analysis scripts refer to the running task and stopping task cohorts respectively. Though, the scripts effectively do the exact same things.
3. lick_detection.py allows the user to check each lick detection session individually for sensor readings and lick peak detection.
4. across_cohort_lick.py and lick_analysis.py are the main analysis scripts for the characterization cohorts' lick data. These scripts work woth lick counts, frontloading metrics, fecal count, and lick bout counts.
