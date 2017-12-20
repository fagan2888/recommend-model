<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkfdMarketreport extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_skfd_marketreport', function($table) {
	    $table->increments('id');
	    $table->string('fd_code')->comment('基金代码');
	    $table->date('report_date')->default('0000-00-00')->comment('报告时间');
	    $table->date('publish_date')->default('0000-00-00')->comment('公布时间');
	    $table->decimal('size',13,0)->comment('基金规模');
	    $table->decimal('hold',5,2)->comment('机构持仓比例%');
	    $table->timestamps();
	});
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('mf_skfd_marketreport');
    }
}
