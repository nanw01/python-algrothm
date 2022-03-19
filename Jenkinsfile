pipeline {
  agent any
  stages {
    stage('master') {
      parallel {
        stage('master') {
          steps {
            echo 'jenkins pipeline'
          }
        }

        stage('platform 1-A') {
          steps {
            fileExists '65465465a.py'
            echo '213'
          }
        }

      }
    }

    stage('test') {
      steps {
        echo 'test'
      }
    }

  }
}