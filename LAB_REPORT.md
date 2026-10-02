# Lab 16 — GCP CPU/LightGBM report

1. Hạ tầng chạy trên project `data-eye-510409-j5`, VM `e2-medium` Debian 12 tại `us-central1-a`, truy cập qua IAP/OS Login.
2. Dataset Credit Card Fraud Detection có 284.807 giao dịch, 30 đặc trưng và 492 giao dịch gian lận; dữ liệu được chia stratified 80/20.
3. Thời gian load dữ liệu là 2.758 s; thời gian training LightGBM với 200 estimators là 7.525 s. Run này không bật early stopping nên `best_iteration` là iteration limit 200, không phải giá trị được tuning.
4. Mô hình đạt AUC-ROC 0,9154 và accuracy 0,9837 trên tập test.
5. F1-score là 0,1550, precision là 0,0851 và recall là 0,8673.
6. Inference một dòng có median latency khoảng 1,275 ms (mean 1,300 ms).
7. Throughput khi dự đoán 1.000 dòng đạt khoảng 113.355 dòng/giây trên CPU.
8. Recall cao nhưng precision/F1 còn thấp do dữ liệu mất cân bằng; có thể cải thiện bằng tuning threshold hoặc tối ưu model.
9. Project đã bật Billing và còn free-trial credits; báo cáo chi phí có thể cập nhật trễ nên đã lưu snapshot Billing hiện tại.
10. Sau khi chụp đủ bằng chứng, cần chạy `terraform destroy` để dừng VM, NAT, Load Balancer và các tài nguyên liên quan.
