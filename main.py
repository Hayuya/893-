import numpy as np
import pandas as pd


employees_df = pd.read_csv('employees.csv')
drivers_df = pd.read_csv('drivers.csv')

working_employees = input("その日出勤する従業員の名前をカンマ区切りで入力してください: ").split(',')
working_drivers = input("その日出勤するドライバーの名前をカンマ区切りで入力してください: ").split(',')

selected_employees_df = employees_df[employees_df['名前'].isin(working_employees)].reset_index(drop=True)
selected_drivers_df = drivers_df[drivers_df['名前'].isin(working_drivers)].reset_index(drop=True)

selected_employees_df['実質乗車人数'] = selected_employees_df['子供の人数'].apply(lambda x: (x // 3) * 2 + (x % 3)) + 1

employee_locations = selected_employees_df[['X座標', 'Y座標']].to_numpy()
driver_locations = selected_drivers_df[['X座標', 'Y座標']].to_numpy()

# 距離行列の計算
distance_matrix_drivers = np.linalg.norm(driver_locations[:, None, :] - employee_locations[None, :, :], axis=2)

# 初期割り当て: 各従業員を最も近いドライバーに割り当て
initial_assignments = np.argmin(distance_matrix_drivers, axis=0)

driver_loads = np.zeros(len(selected_drivers_df), dtype=np.int)
for i, driver_index in enumerate(initial_assignments):
    driver_loads[driver_index] += selected_employees_df.iloc[i]['実質乗車人数']

assignments = [[] for _ in range(len(selected_drivers_df))]
driver_capacities = selected_drivers_df['車の積載人数'].values
available_seats = driver_capacities.copy()

# 全従業員を対象に割り当てを試みる
for employee_index, _ in enumerate(selected_employees_df):
    distances_to_employee = distance_matrix_drivers[:, employee_index]
    driver_order = np.argsort(distances_to_employee)
    
    for driver_index in driver_order:
        if available_seats[driver_index] >= selected_employees_df.iloc[employee_index]['実質乗車人数']:
            assignments[driver_index].append(employee_index)
            available_seats[driver_index] -= selected_employees_df.iloc[employee_index]['実質乗車人数']
            break

# 全従業員が割り当てられているかのチェック
assigned_employee_indices_flat = [item for sublist in assignments for item in sublist]
all_assigned = len(assigned_employee_indices_flat) == len(selected_employees_df)

if not all_assigned:
    unassigned_employees = set(range(len(selected_employees_df))) - set(assigned_employee_indices_flat)
    for employee_index in unassigned_employees:
        distances = distance_matrix_drivers[:, employee_index]
        for driver_index in np.argsort(distances):
            if available_seats[driver_index] >= selected_employees_df.iloc[employee_index]['実質乗車人数']:
                assignments[driver_index].append(employee_index)
                available_seats[driver_index] -= selected_employees_df.iloc[employee_index]['実質乗車人数']
                break

# 結果の出力
for driver_index, driver in selected_drivers_df.iterrows():
    if assignments[driver_index]:
        assigned_employee_names = selected_employees_df.iloc[assignments[driver_index]]['名前'].values
        print(f"ドライバー {driver['名前']} が担当する従業員: {', '.join(assigned_employee_names)}")
    else:
        print(f"ドライバー {driver['名前']} は従業員を割り当てられませんでした。")
