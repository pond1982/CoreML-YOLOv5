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
            request.imageCropAndScaleOption = .scaleFit
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
    
    // SAHI tuning parameters (can be adjusted via UI)
    var sahiTileWidth: CGFloat = 640
    var sahiTileHeight: CGFloat = 640
    var sahiOverlap: CGFloat = 0.15          // 0.0 ... <1.0
    var sahiIoUThreshold: CGFloat = 0.5     // 0.0 ... 1.0
    var sahiScoreThreshold: Float = 0.05    // 0.0 ... 1.0
    var sahiMaxTilesPerFrame: Int = 16
    
    // Run detection on the full frame (no tiling). Handles both VNRecognizedObjectObservation and VNDetectedObjectObservation.
    func detectFullFrame(ciImage: CIImage,
                         iouThreshold: CGFloat,
                         scoreThreshold: Float) -> [Detection] {
        guard let coreMLRequest = coreMLRequest else { return [] }
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        do {
            try handler.perform([coreMLRequest])
            var detections: [Detection] = []
            if let results = coreMLRequest.results as? [VNRecognizedObjectObservation] {
                for r in results {
                    let flipped = CGRect(x: r.boundingBox.minX,
                                         y: 1 - r.boundingBox.maxY,
                                         width: r.boundingBox.width,
                                         height: r.boundingBox.height)
                    let box = VNImageRectForNormalizedRect(flipped,
                                                           Int(ciImage.extent.width),
                                                           Int(ciImage.extent.height))
                    let confidence = r.confidence
                    if confidence >= scoreThreshold {
                        let label = r.labels.first?.identifier ?? "golfball"
                        detections.append(Detection(box: box, confidence: confidence, label: label, color: .red))
                    }
                }
            } else if let results = coreMLRequest.results as? [VNDetectedObjectObservation] {
                for r in results {
                    let flipped = CGRect(x: r.boundingBox.minX,
                                         y: 1 - r.boundingBox.maxY,
                                         width: r.boundingBox.width,
                                         height: r.boundingBox.height)
                    let box = VNImageRectForNormalizedRect(flipped,
                                                           Int(ciImage.extent.width),
                                                           Int(ciImage.extent.height))
                    let confidence = r.confidence
                    if confidence >= scoreThreshold {
                        detections.append(Detection(box: box, confidence: confidence, label: "golfball", color: .red))
                    }
                }
            }
            return nonMaxSuppression(detections, iouThreshold: iouThreshold, scoreThreshold: scoreThreshold)
        } catch {
            print("Full-frame detect error: \(error)")
            return []
        }
    }
    
    // MARK: - SAHI (Slicing Aided Hyper Inference)
    // Tile the image into overlapping crops (no rescale). Each tile’s extent is reset to start at (0,0)
    // so Vision sees a clean coordinate space; we keep the original origin to map boxes back.
    func makeTiles(for image: CIImage,
                   tileSize: CGSize = CGSize(width: 640, height: 640),
                   overlap: CGFloat = 0.2) -> [(image: CIImage, origin: CGPoint)] {
        precondition(overlap >= 0 && overlap < 1, "overlap must be in [0, 1)")
        let stepX = max(1, Int((tileSize.width * (1 - overlap)).rounded()))
        let stepY = max(1, Int((tileSize.height * (1 - overlap)).rounded()))

        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        var tiles: [(CIImage, CGPoint)] = []

        var y = 0
        while y < height {
            var x = 0
            while x < width {
                let w = min(Int(tileSize.width), width - x)
                let h = min(Int(tileSize.height), height - y)
                if w <= 0 || h <= 0 { break }

                let rect = CGRect(x: x, y: y, width: w, height: h)
                let tile = image
                    .cropped(to: rect)
                    .transformed(by: CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y))
                tiles.append((tile, CGPoint(x: rect.origin.x, y: rect.origin.y)))

                x += stepX
            }
            y += stepY
        }
        return tiles
    }

    func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    func nonMaxSuppression(_ dets: [Detection],
                           iouThreshold: CGFloat,
                           scoreThreshold: Float) -> [Detection] {
        var filtered = dets.filter { $0.confidence >= scoreThreshold }
        filtered.sort { $0.confidence > $1.confidence }

        var keep: [Detection] = []
        while let first = filtered.first {
            keep.append(first)
            filtered.removeFirst()
            filtered = filtered.filter { iou(first.box, $0.box) < iouThreshold }
        }
        return keep
    }

    // Run the Core ML + Vision request on each tile and map boxes back to full-image coordinates,
    // then merge with NMS. Returns final detections in full-image coordinate space.
    func detectOnTiles(ciImage: CIImage,
                       tileSize: CGSize = CGSize(width: 640, height: 640),
                       overlap: CGFloat = 0.2,
                       iouThreshold: CGFloat = 0.5,
                       scoreThreshold: Float = 0.20) -> [Detection] {
        guard let coreMLRequest = coreMLRequest else { return [] }

        // If the frame is smaller than a single tile, just run full-frame detection
        if ciImage.extent.width <= tileSize.width && ciImage.extent.height <= tileSize.height {
            return detectFullFrame(ciImage: ciImage, iouThreshold: iouThreshold, scoreThreshold: scoreThreshold)
        }

        // Generate tiles and cap the total number for performance
        var tiles = makeTiles(for: ciImage, tileSize: tileSize, overlap: overlap)
        if tiles.count > sahiMaxTilesPerFrame {
            // Increase tile size to reduce the number of tiles, and reduce overlap for speed
            let scale = ceil(sqrt(Double(tiles.count) / Double(max(1, sahiMaxTilesPerFrame))))
            let newWidth = min(ciImage.extent.width, tileSize.width * CGFloat(scale))
            let newHeight = min(ciImage.extent.height, tileSize.height * CGFloat(scale))
            tiles = makeTiles(for: ciImage,
                              tileSize: CGSize(width: newWidth, height: newHeight),
                              overlap: 0.1)
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.messageLabel.isHidden = false
            self?.messageLabel.text = "SAHI tiles: \(tiles.count) (tile=\(Int(tileSize.width))x\(Int(tileSize.height)), ov=\(String(format: "%.2f", overlap)))"
        }

        var allDetections: [Detection] = []
        for (tileImage, origin) in tiles {
            let handler = VNImageRequestHandler(ciImage: tileImage, options: [:])
            do {
                try handler.perform([coreMLRequest])

                if let results = coreMLRequest.results as? [VNRecognizedObjectObservation] {
                    for r in results {
                        let flipped = CGRect(x: r.boundingBox.minX,
                                             y: 1 - r.boundingBox.maxY,
                                             width: r.boundingBox.width,
                                             height: r.boundingBox.height)
                        let inTile = VNImageRectForNormalizedRect(flipped,
                                                                  Int(tileImage.extent.width),
                                                                  Int(tileImage.extent.height))
                        let inFull = inTile.offsetBy(dx: origin.x, dy: origin.y)
                        let label = r.labels.first?.identifier ?? "golfball"
                        allDetections.append(Detection(box: inFull, confidence: r.confidence, label: label, color: .red))
                    }
                } else if let results = coreMLRequest.results as? [VNDetectedObjectObservation] {
                    for r in results {
                        let flipped = CGRect(x: r.boundingBox.minX,
                                             y: 1 - r.boundingBox.maxY,
                                             width: r.boundingBox.width,
                                             height: r.boundingBox.height)
                        let inTile = VNImageRectForNormalizedRect(flipped,
                                                                  Int(tileImage.extent.width),
                                                                  Int(tileImage.extent.height))
                        let inFull = inTile.offsetBy(dx: origin.x, dy: origin.y)
                        allDetections.append(Detection(box: inFull, confidence: r.confidence, label: "golfball", color: .red))
                    }
                }
            } catch {
                print("Tile detect error: \(error)")
            }
        }

        return nonMaxSuppression(allDetections, iouThreshold: iouThreshold, scoreThreshold: scoreThreshold)
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
                                // First try full-frame (faster). If empty, fallback to SAHI.
                                let full = safeSelf.detectFullFrame(ciImage: ciImage,
                                                                    iouThreshold: safeSelf.sahiIoUThreshold,
                                                                    scoreThreshold: safeSelf.sahiScoreThreshold)
                                if !full.isEmpty {
                                    return (ciImage, full)
                                }
                                let tileSize = CGSize(width: safeSelf.sahiTileWidth, height: safeSelf.sahiTileHeight)
                                let sahi = safeSelf.detectOnTiles(ciImage: ciImage,
                                                                  tileSize: tileSize,
                                                                  overlap: safeSelf.sahiOverlap,
                                                                  iouThreshold: safeSelf.sahiIoUThreshold,
                                                                  scoreThreshold: safeSelf.sahiScoreThreshold)
                                return (ciImage, sahi)
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

