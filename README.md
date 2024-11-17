# ComfyUI_EasyTrain

![1step_](https://github.com/user-attachments/assets/53229cca-46af-4cb8-a978-390d1007442a)
![middle](https://github.com/user-attachments/assets/ddffa32c-30f4-42e4-9585-8bd748e4b586)

***ComfyUI를 이용한 모델 학습용 custom node입니다.

***test 버전이기에 거의 대부분의 기능이 지원되지 않습니다.

***기본적인 모듈과 model save, model load 및 dataloader을 우선적으로 구현할 예정입니다.

원하는 epoch를 설정한 후, Auto queue를 통해 학습을 진행합니다.
학습이 완료되면 exception이 발생하며 queue가 종료됩니다.

CustomModelTrainer node의 accumulate 값은 Gradient accumulation 설정입니다.
필요하지 않은 경우 1로 설정해주십시오.

주의 : 학습 도중 특정 노드가 비활성화 될 경우, 해당 노드의 input에
"trigger":("INT", {"default":0})를 추가한 후 해당 노드의 실행 함수에
self.trigger=trigger 을 추가하여주십시오. 이 후 추가된 trigger 입력에
primitive 노드를 연결한 후 control_after_generate : randomize로 설정하여 주십시오.

권장 파이토치 버전 : pytorch 2.5.1 + cu121
