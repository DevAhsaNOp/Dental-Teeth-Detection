import jinja2
import pdfkit
import random
import string
from datetime import datetime

global patientGetDetails


def GeneratePDF(patientDetails, uploadedTestedImages):
    global patientGetDetails
    patientGetDetails = patientDetails
    PatientID = str(patientGetDetails["PT_PatientID"])
    Firstname = str(patientGetDetails["tblPatient"]["P_FirstName"])
    Lastname = str(patientGetDetails["tblPatient"]["P_LastName"])
    PhoneNumber = str(patientGetDetails["tblPatient"]["P_PhoneNumber"])
    EmailAddress = str(patientGetDetails["tblPatient"]["P_Email"])
    Gender = str(patientGetDetails["tblPatient"]["P_Gender"])
    Address = str(patientGetDetails["PatientAddress"]["Area"]) + ', ' + str(
        patientGetDetails["PatientAddress"]["City"]) + ', ' + str(patientGetDetails["PatientAddress"]["State"])
    Age = "22"
    DentistName = "Muhammad Shahzad"
    DentistPhoneNumber = "03312398777"
    Date = datetime.today().strftime("%d %b, %Y")
    Time = datetime.today().strftime("%I:%M:%S %p")
    patientImage = "https://dmswebapp.azurewebsites.net" + str(patientGetDetails["tblPatient"]["P_ProfileImage"]).\
        replace("~", "")
    DentistService = "Smelly Breath and Toothache"
    image_paths = uploadedTestedImages
    image_tags = ''
    for path in image_paths:
        image_tags += f'<img class="image-item" src="{path}" />'

    context = {'PatientID': PatientID, 'Firstname': Firstname, 'Lastname': Lastname, 'PhoneNumber': PhoneNumber,
               'EmailAddress': EmailAddress, 'Gender': Gender, 'Address': Address, 'Age': Age, 'DentistName': DentistName,
               'DentistPhoneNumber': DentistPhoneNumber, 'Date': Date, 'patientImage': patientImage,
               'DentistService': DentistService, 'Time': Time, 'image_tags': image_tags}

    template_loader = jinja2.FileSystemLoader('./')
    template_env = jinja2.Environment(loader=template_loader)

    html_template = 'index.html'
    template = template_env.get_template(html_template)
    output_text = template.render(context)

    options = {
        'margin-top': '0',
        'margin-right': '0',
        'margin-bottom': '0',
        'margin-left': '0'
    }

    config = pdfkit.configuration(wkhtmltopdf='.\wkhtmltopdf\\bin\\wkhtmltopdf.exe')
    randomPDFName = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
    output_pdf = 'Patient' + str(patientGetDetails["PT_PatientID"]) + '_' + randomPDFName + '.pdf'
    pdfkit.from_string(output_text, output_pdf, configuration=config, css='style.css', options=options)
