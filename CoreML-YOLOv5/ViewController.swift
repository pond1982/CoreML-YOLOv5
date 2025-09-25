//
//  ViewController.swift
//  CoreML-YOLOv5
//
//  Created by DAISUKE MAJIMA on 2021/12/28.


import UIKit
import PhotosUI
import Vision
import AVKit

class ViewController: UIViewController, PHPickerViewControllerDelegate {

    @IBOutlet weak var messageLabel: UILabel!
    
    var spinner: UIActivityIndicatorView!
    
    lazy var coreMLRequest:VNCoreMLRequest? = {
        do {
            let model = try golfballyolov8n(configuration: MLModelConfiguration()).model
            let vnCoreMLModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnCoreMLModel)
            request.imageCropAndScaleOption = .scaleFill
            return request
        } catch let error {
            print(error)
            return nil
        }
    }()
    
    let classLabels = ["golfball"]
    
    let colorSet:[UIColor] = {
        var colorSet:[UIColor] = []

        for _ in 0...80 {
            let color = UIColor(red: CGFloat.random(in: 0...1), green: CGFloat.random(in: 0...1), blue: CGFloat.random(in: 0...1), alpha: 1)
            colorSet.append(color)
        }
        
        return colorSet
    }()
    
    var ciContext = CIContext()
    
    enum MediaMode {
        case photo
        case video
    }
    
    var mediaMode:MediaMode = .video

    var initializeTimer:Timer?
    var frame = 0

    @IBOutlet weak var imageView: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        spinner = UIActivityIndicatorView(style: .large)
        spinner.center = view.center
        spinner.hidesWhenStopped = true
        view.addSubview(spinner)
        presentPhPicker()
    }
    
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        messageLabel.isHidden = true
        switch mediaMode {
        case .photo:
            guard let result = results.first else { return }
            if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
                result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] image, error  in
                    if let image = image as? UIImage,  let safeSelf = self {
                        
                        let correctOrientImage = safeSelf.getCorrectOrientationUIImage(uiImage: image) // iPhoneのカメラで撮った画像は回転している場合があるので画像の向きに応じて補正
                        
                        // モデルの初期化が終わっているか確認して検出実行
                        if self?.coreMLRequest != nil {
                            safeSelf.detect(image: correctOrientImage)
                        } else {
                            self?.initializeTimer = Timer(timeInterval: 0.5, repeats: true, block: { timer in
                                if self?.coreMLRequest != nil {
                                    safeSelf.detect(image: correctOrientImage)
                                    timer.invalidate()
                                }
                            })
                        }
                    }
                }
            }
            
        case .video:
            guard let result = results.first else { return }
            guard let typeIdentifier = result.itemProvider.registeredTypeIdentifiers.first else { return }
            if result.itemProvider.hasItemConformingToTypeIdentifier(typeIdentifier) {
                DispatchQueue.main.async { [weak self] in
                    self?.spinner.startAnimating()
                    self?.spinner.isHidden = false
                }
                result.itemProvider.loadFileRepresentation(forTypeIdentifier: typeIdentifier) { [weak self] (url, error) in
                    if let error = error { print("*** error: \(error)") }
                    let start = Date()
                    DispatchQueue.main.async {
                        self?.imageView.image = nil
                    }
                    if let url = url as? URL {
                        result.itemProvider.loadItem(forTypeIdentifier: typeIdentifier, options: nil) { (url, error) in
                            let procceessed = self?.applyProcessingOnVideo(videoURL: url as! URL, { ciImage in
                                guard let safeSelf = self else { return nil }
                                guard let coreMLRequest = safeSelf.coreMLRequest else { return nil }
                                let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
                                do {
                                    try handler.perform([coreMLRequest])
                                    guard let detectResults = coreMLRequest.results as? [VNDetectedObjectObservation] else { return (ciImage, []) }
                                    var detections: [Detection] = []
                                    for detectResult in detectResults {
                                        let flippedBox = CGRect(x: detectResult.boundingBox.minX, y: 1 - detectResult.boundingBox.maxY, width: detectResult.boundingBox.width, height: detectResult.boundingBox.height)
                                        let box = VNImageRectForNormalizedRect(flippedBox, Int(ciImage.extent.width), Int(ciImage.extent.height))
                                        let confidence = detectResult.confidence
                                        let label = "golfball"
                                        let detection = Detection(box: box, confidence: confidence, label: label, color: .red)
                                        detections.append(detection)
                                    }
                                    if let newImage = safeSelf.drawRectOnImage(detections, ciImage) {
                                        return (newImage.resize(as: ciImage.extent.size), detections)
                                    }
                                    return (ciImage, detections)
                                } catch {
                                    print(error)
                                    return (ciImage, [])
                                }
                            }, { err, processedVideoURL in
                                let end = Date()
                                let diff = end.timeIntervalSince(start)
                                print(diff)
                                let player = AVPlayer(url: processedVideoURL!)
                                DispatchQueue.main.async {
                                    self?.messageLabel.isHidden = true
                                    self?.spinner.stopAnimating()
                                    let controller = AVPlayerViewController()
                                    controller.player = player
                                    self?.present(controller, animated: true) {
                                        player.play()
                                    }
                                }
                            })
                        }
                    }
                }
            }
        }
    }
    
    @IBAction func selectImageButtonTapped(_ sender: UIButton) {
        presentPhPicker()
    }
    
    @IBAction func saveButtonTapped(_ sender: UIButton) {
        UIImageWriteToSavedPhotosAlbum(imageView.image!, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            // we got back an error!
            let ac = UIAlertController(title: "Save error", message: error.localizedDescription, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        } else {
            let ac = UIAlertController(title: NSLocalizedString("saved!",value: "saved!", comment: ""), message: NSLocalizedString("Saved in photo library",value: "Saved in photo library", comment: ""), preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
    }
    
    func presentPhPicker(){
        let alert = UIAlertController(title: "Select Video", message: "", preferredStyle: .actionSheet)
        
        let videoAction = UIAlertAction(title: "Video", style: .default) { action in
            var configuration = PHPickerConfiguration()
            configuration.selectionLimit = 1
            configuration.filter = .videos
            let picker = PHPickerViewController(configuration: configuration)
            picker.delegate = self
            self.mediaMode = .video
            self.present(picker, animated: true)
        }
        alert.addAction(videoAction)
        self.present(alert, animated: true)
    }
    
    func getCorrectOrientationUIImage(uiImage:UIImage) -> UIImage {
        var newImage = UIImage()
        let ciContext = CIContext()
        switch uiImage.imageOrientation.rawValue {
        case 1:
            guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.down),
                  let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
            
            newImage = UIImage(cgImage: cgImage)
        case 3:
            guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.right),
                  let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
            newImage = UIImage(cgImage: cgImage)
        default:
            newImage = uiImage
        }
        return newImage
    }
}

struct Detection {
    let box:CGRect
    let confidence:Float
    let label:String?
    let color:UIColor
}

