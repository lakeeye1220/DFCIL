import matplotlib.pyplot as plt

# 데이터
task_num = 5
x = [i for i in range(task_num)]

if task_num ==5:
    loss_spkd = [0, 0.000242788, 3.22E-05, 4.92E-05, 0.000399752]
    loss_rkd = [0, 3.825860381, 2.973817468, 2.410305411, 2.071820028]
elif task_num == 10:
    loss_spkd = [0, 8.02E-05, 2.59E-05,	2.93E-05,	3.28E-05,	5.36E-06,	9.22E-06,	1.39E-05,	7.40E-05,	5.81E-05]
    loss_rkd = [0,2.917124212,2.527541786,	1.723622441,1.572469205,1.396741837,1.247971624,1.246666968,1.302376255,1.380933493]
else:
    loss_spkd = []
    loss_rkd = []

# 그래프 생성
plt.figure(figsize=(8, 5))

# 꺾은 선 그래프 그리기
plt.plot(x, loss_spkd, marker='*', markersize=8,label='SPKD Loss',color='crimson')
plt.plot(x, loss_rkd, marker='o',markersize=8, label='RKD loss',color='cornflowerblue')

# 각 꺾인 부분에 loss 값 표시
for i in range(len(x)):
    if i>=1:
        plt.text(x[i], loss_spkd[i], f'{loss_spkd[i]:.1e}', ha='center', va='bottom',size=9)
        plt.text(x[i], loss_rkd[i], f'{loss_rkd[i]:.3f}', ha='center', va='bottom',size=9)
    else:
        plt.text(x[i], loss_spkd[i], f'{loss_spkd[i]:.1e}', ha='center', va='bottom',size=9)

# 레이블 및 제목 설정
plt.xlabel('Task Index',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.title('Comparison of SPKD and RKD Loss on ImageNet-100 {}tasks'.format(task_num),fontsize=14)
plt.legend(fontsize=12)
plt.grid(True,linestyle='dotted', color='gray')
plt.xticks(x)

plt.savefig('./{}task_ImageNet-100_spkd_rkd_loss.pdf'.format(task_num))

# 그래프 출력
plt.show()